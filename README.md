# Vocal Lens Backend

Runpod Serverless worker for low-cost vocal practice analysis.

## What it returns

- BPM and estimated key
- practical vocal range
- pitch contour
- suggested breathing points
- register-transition practice candidates

The last two values are heuristics for practice support, not professional or medical diagnoses.

## Input

```json
{
  "input": {
    "audio_url": "https://example.com/temporary-upload.wav"
  }
}
```

The URL must point to a user-authorized audio file. Files are capped at 30 MB / 8 minutes and deleted after each job.

## Runpod settings

- Worker type: Serverless Flex
- Active workers: `0`
- Max workers: `1`
- Idle timeout: `5 seconds`
- GPU: not required for this first low-cost worker; choose the least expensive compatible worker

Keeping active workers at zero prevents idle charges. A later version can add GPU stem separation after usage is validated.

## Security

Never commit `RUNPOD_API_KEY` or other credentials. Keep secrets in Runpod and the frontend hosting environment.
