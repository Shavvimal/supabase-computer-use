import requests
import base64
import json

# Endpoint URL
url = "http://localhost:8000/agent"

# Read an image file and encode it in base64
with open("./img.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Construct the payload
payload = {
    "content_blocks": [
        {
            "type": "text",
            "content": {
                "text": "Create a new table called emails."
            }
        },
        {
            "type": "image",
            "content": {
                "source": {
                    "type": "base64",
                    "data": f"data:image/png;base64,{encoded_image}"
                }
            }
        }
    ],
    "metadata": {
        "screen": {
            "width": 1512,
            "height": 982,
            "availWidth": 1512,
            "availHeight": 888,
            "colorDepth": 30,
            "pixelDepth": 30,
            "orientation": "landscape-primary"
        },
        "window": {
            "innerWidth": 857,
            "innerHeight": 868,
            "outerWidth": 1512,
            "outerHeight": 888,
            "devicePixelRatio": 2
        },
        "browser": {
            "userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/131.0.0.0 Safari/537.36",
            "platform": "MacIntel",
            "language": "en-US",
            "languages": [
                "en-US",
                "en"
            ],
            "cookieEnabled": True,
            "doNotTrack": "1",
            "vendor": "Google Inc.",
            "maxTouchPoints": 0
        },
        "connection": {
            "effectiveType": "4g",
            "downlink": 10,
            "rtt": 50
        }
    }
}

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Send the POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Check if the request was successful
if response.status_code == 200:
    # Parse and print the response
    response_data = response.json()
    print(json.dumps(response_data, indent=2))
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)
