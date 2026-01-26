#!/bin/bash
set -e

echo "=============================================="
echo "LLM Safety Study - AWS Infrastructure Setup"
echo "=============================================="
echo ""

# ============================================
# CONFIGURATION
# ============================================
# Required environment variables:
#   NOTIFICATION_EMAIL - Email for alerts
#
# Optional:
#   AWS_PROFILE - AWS CLI profile to use (default: uses default profile)
#
# Prerequisites:
#   1. Create IAM user 'llm-safety-study-admin' in AWS Console
#   2. Attach 'LLMSafetyStudyAdminPolicy' (see llm-safety-admin-policy.json)
#   3. Create access keys and configure AWS CLI:
#      aws configure --profile llm-safety
#
# Usage:
#   export AWS_PROFILE=llm-safety
#   export NOTIFICATION_EMAIL=you@example.com
#   ./setup_all.sh
# ============================================

EMAIL="${NOTIFICATION_EMAIL:?ERROR: Set NOTIFICATION_EMAIL environment variable}"
REGION="us-east-1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# AWS CLI command with profile support
if [ -n "$AWS_PROFILE" ]; then
    AWS_CMD="aws --profile $AWS_PROFILE"
    echo "Using AWS profile: $AWS_PROFILE"
else
    AWS_CMD="aws"
    echo "Using default AWS credentials"
fi

# Cleanup function for temp files
cleanup() {
    rm -f "$SCRIPT_DIR/bedrock-trust-policy-temp.json"
}
trap cleanup EXIT

# ============================================
# CREDENTIAL VERIFICATION
# ============================================
echo ""
echo "Verifying AWS credentials..."

CALLER_IDENTITY=$($AWS_CMD sts get-caller-identity 2>&1) || {
    echo ""
    echo "ERROR: AWS credentials not configured or invalid."
    echo ""
    echo "Setup steps:"
    echo "  1. Create IAM user 'llm-safety-study-admin' in AWS Console"
    echo "  2. Attach 'LLMSafetyStudyAdminPolicy' (see llm-safety-admin-policy.json)"
    echo "  3. Create access keys for the user"
    echo "  4. Run: aws configure --profile llm-safety"
    echo "  5. Run this script with: AWS_PROFILE=llm-safety ./setup_all.sh"
    exit 1
}

# Extract account ID from caller identity
ACCOUNT_ID=$(echo "$CALLER_IDENTITY" | grep -o '"Account": "[0-9]*"' | grep -o '[0-9]*')
CALLER_ARN=$(echo "$CALLER_IDENTITY" | grep -o '"Arn": "[^"]*"' | cut -d'"' -f4)

echo "  Account ID: $ACCOUNT_ID"
echo "  Caller ARN: $CALLER_ARN"

# Warn if running as root
if echo "$CALLER_ARN" | grep -q ":root$"; then
    echo ""
    echo "WARNING: Running as root user is not recommended."
    echo "         Use an IAM user with LLMSafetyStudyAdminPolicy instead."
    echo ""
    read -p "Continue anyway? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "Configuration:"
echo "  Account ID: $ACCOUNT_ID"
echo "  Email: $EMAIL"
echo "  Region: $REGION"
echo ""

# ============================================
# PREPARE TRUST POLICY (IDEMPOTENT)
# ============================================
cp "$SCRIPT_DIR/bedrock-trust-policy.json" "$SCRIPT_DIR/bedrock-trust-policy-temp.json"

# macOS and Linux compatible sed
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/ACCOUNT_ID_PLACEHOLDER/$ACCOUNT_ID/g" "$SCRIPT_DIR/bedrock-trust-policy-temp.json"
else
    sed -i "s/ACCOUNT_ID_PLACEHOLDER/$ACCOUNT_ID/g" "$SCRIPT_DIR/bedrock-trust-policy-temp.json"
fi

# ============================================
# STEP 1: Create Bedrock Service Role
# ============================================
echo "Step 1/6: Creating Bedrock service role..."

$AWS_CMD iam create-role \
    --role-name BedrockModelImportRole \
    --assume-role-policy-document "file://$SCRIPT_DIR/bedrock-trust-policy-temp.json" 2>/dev/null || echo "  Role may already exist"

$AWS_CMD iam put-role-policy \
    --role-name BedrockModelImportRole \
    --policy-name BedrockS3Access \
    --policy-document "file://$SCRIPT_DIR/bedrock-s3-policy.json" 2>/dev/null || echo "  Role policy may already exist"

# ============================================
# STEP 2: Create S3 Bucket
# ============================================
echo "Step 2/6: Creating S3 bucket..."

BUCKET_NAME="llm-safety-study-weights-${ACCOUNT_ID}"

$AWS_CMD s3 mb "s3://${BUCKET_NAME}" --region "$REGION" 2>/dev/null || echo "  Bucket may already exist"

echo "  Bucket: $BUCKET_NAME"

# ============================================
# STEP 3: Create SNS Topic
# ============================================
echo "Step 3/6: Creating SNS notification topic..."

TOPIC_ARN=$($AWS_CMD sns create-topic --name llm-safety-study-alerts --query 'TopicArn' --output text)
echo "  Topic ARN: $TOPIC_ARN"

$AWS_CMD sns subscribe \
    --topic-arn "$TOPIC_ARN" \
    --protocol email \
    --notification-endpoint "$EMAIL"

echo ""
echo ">>> ACTION REQUIRED: Check your email and confirm the SNS subscription <<<"
echo ">>> Press Enter after confirming the subscription... <<<"
read -r

# ============================================
# STEP 4: Create Cost Alarms
# ============================================
echo "Step 4/6: Creating cost alarms..."

for THRESHOLD in 75 150 250; do
    $AWS_CMD cloudwatch put-metric-alarm \
        --alarm-name "llm-safety-study-cost-${THRESHOLD}" \
        --alarm-description "LLM Safety Study costs exceeded \$${THRESHOLD}" \
        --metric-name EstimatedCharges \
        --namespace AWS/Billing \
        --statistic Maximum \
        --period 21600 \
        --evaluation-periods 1 \
        --threshold "$THRESHOLD" \
        --comparison-operator GreaterThanThreshold \
        --dimensions Name=Currency,Value=USD \
        --alarm-actions "$TOPIC_ARN" \
        --treat-missing-data notBreaching
    echo "  Created alarm for \$${THRESHOLD}"
done

# ============================================
# STEP 5: Create EventBridge Rule for Bedrock Jobs
# ============================================
echo "Step 5/6: Creating Bedrock job notifications..."

$AWS_CMD events put-rule \
    --name llm-safety-study-import-complete \
    --description "Notify when Bedrock model import jobs complete" \
    --event-pattern '{
        "source": ["aws.bedrock"],
        "detail-type": ["Bedrock Model Import Job State Change"],
        "detail": {
            "status": ["Completed", "Failed"]
        }
    }' \
    --state ENABLED

$AWS_CMD sns add-permission \
    --topic-arn "$TOPIC_ARN" \
    --label AllowEventBridgePublish \
    --aws-account-id "$ACCOUNT_ID" \
    --action-name Publish 2>/dev/null || echo "  Permission may already exist"

$AWS_CMD events put-targets \
    --rule llm-safety-study-import-complete \
    --targets "[{
        \"Id\": \"sns-notification\",
        \"Arn\": \"$TOPIC_ARN\",
        \"InputTransformer\": {
            \"InputPathsMap\": {
                \"jobName\": \"$.detail.jobName\",
                \"status\": \"$.detail.status\",
                \"modelName\": \"$.detail.importedModelName\"
            },
            \"InputTemplate\": \"\\\"Bedrock Import Job Update\\\\n\\\\nJob: <jobName>\\\\nModel: <modelName>\\\\nStatus: <status>\\\\n\\\\nCheck console for details.\\\"\"
        }
    }]"

# ============================================
# STEP 6: Create Inference Monitoring Alarms
# ============================================
echo "Step 6/6: Creating inference monitoring alarms..."

$AWS_CMD cloudwatch put-metric-alarm \
    --alarm-name llm-safety-study-inference-errors \
    --alarm-description "Bedrock inference error rate too high" \
    --metric-name InvocationErrors \
    --namespace AWS/Bedrock \
    --statistic Sum \
    --period 300 \
    --evaluation-periods 2 \
    --threshold 50 \
    --comparison-operator GreaterThanThreshold \
    --alarm-actions "$TOPIC_ARN" \
    --treat-missing-data notBreaching

$AWS_CMD cloudwatch put-metric-alarm \
    --alarm-name llm-safety-study-throttling \
    --alarm-description "Bedrock requests being throttled" \
    --metric-name ThrottledRequests \
    --namespace AWS/Bedrock \
    --statistic Sum \
    --period 300 \
    --evaluation-periods 1 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold \
    --alarm-actions "$TOPIC_ARN" \
    --treat-missing-data notBreaching

# ============================================
# COMPLETE
# ============================================
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Resources Created:"
echo "  - Bedrock Import Role: BedrockModelImportRole"
echo "  - S3 Bucket: $BUCKET_NAME"
echo "  - SNS Topic: $TOPIC_ARN"
echo "  - Cost Alarms: \$75, \$150, \$250 thresholds"
echo "  - EventBridge Rule: Bedrock import notifications"
echo "  - Inference Alarms: Error rate and throttling"
echo ""
echo "Console Login:"
echo "  URL: https://${ACCOUNT_ID}.signin.aws.amazon.com/console"
echo "  Username: llm-safety-study-admin"
echo ""
echo "Key Console Pages:"
echo "  Bedrock: https://console.aws.amazon.com/bedrock/home?region=${REGION}#/import-models"
echo "  Alarms:  https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#alarmsV2:"
echo "  Costs:   https://console.aws.amazon.com/cost-management/home#/cost-explorer"
echo ""
echo "Next Steps:"
echo "  1. Verify resources in AWS Console"
echo "  2. Import models using model_import_checklist.md"
echo "  3. You'll receive email notifications when imports complete"
echo ""

# Save configuration for later scripts
cat > "$SCRIPT_DIR/config_output.sh" << CONFIGEOF
export SNS_TOPIC_ARN="$TOPIC_ARN"
export S3_BUCKET="$BUCKET_NAME"
export AWS_ACCOUNT_ID="$ACCOUNT_ID"
export AWS_REGION="$REGION"
CONFIGEOF

echo "Configuration saved to config_output.sh"
echo "Source this file before running other scripts: source config_output.sh"
echo ""
