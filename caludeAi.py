import json
import time
import boto3
from urllib.request import urlopen

def lambda_handler(event, context):
    transcribe = boto3.client("transcribe")
    s3 = boto3.client("s3")
    translate = boto3.client("translate")
    bedrock_runtime = boto3.client("bedrock-runtime", "us-east-1")
    if event:
        file_obj = event["Records"][0]
        bucket_name = str(file_obj["s3"]["bucket"]["name"])
        file_name = str(file_obj["s3"]["object"]["key"])
        s3_uri = create_uri(bucket_name, file_name)
   
        job_name = context.aws_request_id
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            IdentifyLanguage=True,
            LanguageOptions=["en-US", "zh-CN", "es-US", "es-ES", "en-IN", "en-GB","zh-TW","hi-IN","ko-KR","te-IN","vi-VN","th-TH","ja-JP"],  # List of languages to identify
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 2,
            }
        )

        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]
            if job_status in ["COMPLETED", "FAILED"]:
                break
            print(f"Transcription job status: {job_status}")
            time.sleep(10)

        if job_status == "COMPLETED":
            print("Transcription job completed successfully.")
            transcription_results_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            transcription_results = json.load(urlopen(transcription_results_uri))
            processed_result = process_speaker_separation(transcription_results)
            processed_result_str = ". ".join(processed_result["transcripts"])
            translated_processed_result_str  = translate_text(translate, processed_result_str, 'en')
            translated_processed_result = {"transcripts": translated_processed_result_str.split(". ")}
            summary, traits = summarize_transcription_with_claude(translated_processed_result_str, bedrock_runtime)
            
            summary_key = f"transcription_summary/{file_name}.json"
            save_summary_to_s3(s3, bucket_name, summary_key, summary)
            save_results_to_s3(s3, bucket_name, file_name, translated_processed_result)
            save_traits_to_s3(s3, bucket_name, file_name, traits)

            return {
                'statusCode': 200,
                'body': json.dumps('Transcription complete')
            }
        else:
            print(f"Transcription job failed with status: {job_status}")
            return {
                'statusCode': 500,
                'body': json.dumps('Transcription job failed')
            }

def create_uri(bucket_name, file_name):
    return "s3://" + bucket_name + "/" + file_name

def process_speaker_separation(original_json):
    items = original_json["results"]["items"]
    speakers_transcripts = []  # Initialize an empty array to hold each speaker's transcript
    current_speaker = None
    speaker_text = ""  # Initialize an empty string to accumulate the current speaker's text

    for item in items:
        speaker_label = item.get('speaker_label', None)
        content = item["alternatives"][0]["content"]

        # Check if there's a new speaker or if it's the first speaker
        if speaker_label is not None and speaker_label != current_speaker:
            # If it's not the first speaker, append the previous speaker's text to the array
            if current_speaker is not None:
                speakers_transcripts.append(f"{current_speaker}: {speaker_text.strip()}")
                speaker_text = ""  # Reset the speaker_text for the new speaker
            current_speaker = speaker_label

        # Avoid adding a space before punctuation
        if item['type'] == 'punctuation':
            speaker_text = speaker_text.rstrip() + content
        else:
            speaker_text += f"{content} "

    # After the loop, append the last speaker's text if it exists
    if speaker_text:
        speakers_transcripts.append(f"{current_speaker}: {speaker_text.strip()}")

    return {"transcripts": speakers_transcripts}
    
def summarize_transcription_with_claude(processed_result, bedrock_runtime):
    # Convert the processed_result into a single string for the prompt
    #transcription_text = ". ".join([item for item in processed_result["transcripts"]])
    if len(processed_result) > 500:
        # Create the detailed prompt if the transcription text is long enough
        prompt = f"""{processed_result}: Given the following detailed conversation transcript between a patient and a healthcare coordinator,donot mention the gender in the summary, create a concise summary in the form of caller-written notes. The summary should include essential details such as patient concerns, notable symptoms, any mentioned appointments, and overall sentiment. Focus on capturing the key points in a clear, organized, and succinct manner as if jotting down notes during the call. Make it as a paragraph.
        Patient Traits: List the patient's traits identified from the call and assessment, using single words for each trait that capture their personality, behavior, and any notable health-related characteristics."""
        # Prepare the request payload
        kwargs = {
            "modelId": "anthropic.claude-v2",
            "contentType": "application/json",
            "accept": "*/*",
            "body": json.dumps({
                "prompt": f"\n\nHuman: {prompt}\nAssistant:",
                "max_tokens_to_sample": 2000,
                "temperature": 1,
                "top_k": 250,
                "top_p": 0.999,
                "stop_sequences": ["\n\nHuman:"],
                "anthropic_version": "bedrock-2023-05-31"
            })
        }
    
        response = bedrock_runtime.invoke_model(**kwargs)
        resp_body = json.loads(response.get('body').read())
        summary, traits = parse_ai_response(resp_body.get('completion', ''))
        return summary, traits
    else:
        return ("Transcription data insufficient for a meaningful summary. Please check the transcription and ensure it is complete before attempting to summarize again.", "No traits identified due to insufficient data.")

def parse_ai_response(ai_response):
    try:
        summary, traits = ai_response.split("Patient Traits:")
        summary = summary.strip()
        traits = traits.strip()
        return summary, traits
    except ValueError:
        return ai_response, "No traits identified" 
        
def translate_text(translate, text, target_language):
    response = translate.translate_text(
        Text=text,
        SourceLanguageCode='auto',
        TargetLanguageCode=target_language
    )
    return response['TranslatedText']
    
def save_results_to_s3(s3, bucket_name, file_name, processed_result):
    # Save processed results to S3
    s3_key = f"transcription_results/{file_name}.json"
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=json.dumps(processed_result).encode('utf-8'))
    print(f"Transcription results uploaded to S3: s3://{bucket_name}/{s3_key}")

def save_summary_to_s3(s3, bucket_name, s3_key, content):
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=json.dumps(content).encode('utf-8'))
    print(f"Results uploaded to S3: s3://{bucket_name}/{s3_key}")

def save_traits_to_s3(s3, bucket_name, file_name, traits):
    s3_key = f"patient_traits/{file_name}.json"
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=json.dumps({"traits": traits}).encode('utf-8'))
    print(f"Patient traits uploaded to S3: s3://{bucket_name}/{s3_key}")