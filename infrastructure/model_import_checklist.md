# Model Import Checklist

## Prerequisites
- [ ] Run `source config_output.sh` to load environment variables
- [ ] HuggingFace CLI installed: `pip install huggingface_hub`
- [ ] HuggingFace logged in: `huggingface-cli login`
- [ ] Llama model access approved at https://huggingface.co/meta-llama

## Models to Import

### 1. Qwen 3 8B Base
```bash
# Download
huggingface-cli download Qwen/Qwen3-8B-Base --local-dir ./models/qwen3-8b-base

# Upload to S3
aws s3 sync ./models/qwen3-8b-base s3://${S3_BUCKET}/qwen3-8b-base/

# Import to Bedrock
aws bedrock create-model-import-job \
    --job-name qwen3-8b-base-import \
    --imported-model-name qwen3-8b-base \
    --role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockModelImportRole \
    --model-data-source "s3DataSource={s3Uri=s3://${S3_BUCKET}/qwen3-8b-base/}"
```
- [ ] Downloaded
- [ ] Uploaded to S3
- [ ] Import job created
- [ ] Import completed (check email)

### 2. Qwen 3 8B Instruct
```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b-instruct

aws s3 sync ./models/qwen3-8b-instruct s3://${S3_BUCKET}/qwen3-8b-instruct/

aws bedrock create-model-import-job \
    --job-name qwen3-8b-instruct-import \
    --imported-model-name qwen3-8b-instruct \
    --role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockModelImportRole \
    --model-data-source "s3DataSource={s3Uri=s3://${S3_BUCKET}/qwen3-8b-instruct/}"
```
- [ ] Downloaded
- [ ] Uploaded to S3
- [ ] Import job created
- [ ] Import completed

### 3. Llama 3.1 8B Base
**Note: Requires approval from Meta on HuggingFace**
```bash
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir ./models/llama31-8b-base

aws s3 sync ./models/llama31-8b-base s3://${S3_BUCKET}/llama31-8b-base/

aws bedrock create-model-import-job \
    --job-name llama31-8b-base-import \
    --imported-model-name llama31-8b-base \
    --role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockModelImportRole \
    --model-data-source "s3DataSource={s3Uri=s3://${S3_BUCKET}/llama31-8b-base/}"
```
- [ ] HuggingFace access approved
- [ ] Downloaded
- [ ] Uploaded to S3
- [ ] Import job created
- [ ] Import completed

### 4. Llama 3.1 8B Instruct
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/llama31-8b-instruct

aws s3 sync ./models/llama31-8b-instruct s3://${S3_BUCKET}/llama31-8b-instruct/

aws bedrock create-model-import-job \
    --job-name llama31-8b-instruct-import \
    --imported-model-name llama31-8b-instruct \
    --role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockModelImportRole \
    --model-data-source "s3DataSource={s3Uri=s3://${S3_BUCKET}/llama31-8b-instruct/}"
```
- [ ] Downloaded
- [ ] Uploaded to S3
- [ ] Import job created
- [ ] Import completed

### 5. Mistral 7B v0.3 Base
```bash
huggingface-cli download mistralai/Mistral-7B-v0.3 --local-dir ./models/mistral-7b-base

aws s3 sync ./models/mistral-7b-base s3://${S3_BUCKET}/mistral-7b-base/

aws bedrock create-model-import-job \
    --job-name mistral-7b-base-import \
    --imported-model-name mistral-7b-base \
    --role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockModelImportRole \
    --model-data-source "s3DataSource={s3Uri=s3://${S3_BUCKET}/mistral-7b-base/}"
```
- [ ] Downloaded
- [ ] Uploaded to S3
- [ ] Import job created
- [ ] Import completed

### 6. Mistral 7B v0.3 Instruct
```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ./models/mistral-7b-instruct

aws s3 sync ./models/mistral-7b-instruct s3://${S3_BUCKET}/mistral-7b-instruct/

aws bedrock create-model-import-job \
    --job-name mistral-7b-instruct-import \
    --imported-model-name mistral-7b-instruct \
    --role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockModelImportRole \
    --model-data-source "s3DataSource={s3Uri=s3://${S3_BUCKET}/mistral-7b-instruct/}"
```
- [ ] Downloaded
- [ ] Uploaded to S3
- [ ] Import job created
- [ ] Import completed

## Monitoring Commands

Check import job status:
```bash
aws bedrock list-model-import-jobs --query 'modelImportJobSummaries[*].[jobName,status]' --output table
```

List imported models:
```bash
aws bedrock list-imported-models --query 'modelSummaries[*].[modelName,modelArn]' --output table
```

## Test Inference

After all models are imported, test each one:
```bash
# Test Qwen base
aws bedrock-runtime invoke-model \
    --model-id arn:aws:bedrock:us-east-1:${AWS_ACCOUNT_ID}:imported-model/qwen3-8b-base \
    --body '{"prompt": "Hello, how are you?", "max_tokens": 50}' \
    --content-type application/json \
    --accept application/json \
    output.json && cat output.json
```

- [ ] qwen3-8b-base responds
- [ ] qwen3-8b-instruct responds
- [ ] llama31-8b-base responds
- [ ] llama31-8b-instruct responds
- [ ] mistral-7b-base responds
- [ ] mistral-7b-instruct responds

## Troubleshooting

**Import stuck > 2 hours:**
- Check CloudWatch logs: /aws/bedrock/model-import
- Verify S3 permissions on BedrockModelImportRole

**Import failed:**
- Common: Model format not supported (needs safetensors)
- Common: S3 permissions missing
- Check error in: Bedrock Console > Import jobs > [job] > Details

**Inference returns 403:**
- Model not yet active (wait for import completion)
- Verify model ARN is correct
- Check IAM policy includes bedrock:InvokeModel
