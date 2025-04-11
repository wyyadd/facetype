---
title: Facetype
emoji: 🏢
colorFrom: green
colorTo: pink
sdk: gradio
sdk_version: 5.24.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Face Type API

## Endpoint

### `GET /face-type`

Retrieves a unique identifier (`face_type`) for a given face image.  
Docker hub: https://hub.docker.com/r/wyyadd/face-type

## Request Parameters

| Parameter | Type   | Required | Description              |
|-----------|--------|----------|--------------------------|
| `url`     | string | Yes      | Public URL of the image. |

## Response

### Success Response
**Status Code:** `200 OK`  
**Note**: face type is from 0 to 31.

```json
{
  "face_type": 12
}
```

### Error Responses
**Invalid URL or Download Failure**  
**Status Code**: `400 Bad Request`
```json
{
  "detail": "Failed to download image from URL: <error message>"
}
```
**No Face Detected**  
**Status Code**: `500 Internal Server Error`
```json
{
  "detail": "No face detected."
}
```

## Example
```bash
curl -X 'GET' \
  'http://127.0.0.1:80/face-type?url=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2Fe%2Fe9%2FDonald_Trump_NYMA.jpg%2F170px-Donald_Trump_NYMA.jpg' \
  -H 'accept: application/json'
```

