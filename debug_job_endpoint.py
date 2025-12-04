import requests
import json
import sys

job_id = "c29b5038-5761-429a-9b3f-f6cedb976581"
url = f"http://127.0.0.1:8000/jobs/{job_id}"

print(f"Fetching status for job {job_id} from {url}...")

try:
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2))
        
        result = data.get("result")
        if result:
            clips = result.get("clips", [])
            print(f"\nFound {len(clips)} clips in result.")
        else:
            print("\nNo 'result' field in response.")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Request failed: {e}")
