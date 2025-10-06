echo "Gemini Prompt"
API_KEY=AQ.Ab8RN6KG2JLCzn0JUtFnTeK2MK4iJcs1tVxuV7fG_8DvZzq0SA

curl "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:streamGenerateContent?key=$API_KEY" \
-X POST \
-H "Content-Type: application/json" \
-d '{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "how far is the earth from the moon"
        }
      ]
    }
  ]
}'