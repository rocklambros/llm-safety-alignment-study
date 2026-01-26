#!/bin/bash

# Load configuration
source config_output.sh 2>/dev/null || {
    echo "ERROR: Run setup_all.sh first and source config_output.sh"
    exit 1
}

echo "Testing notification pipeline..."
echo "SNS Topic: $SNS_TOPIC_ARN"
echo ""

# Test 1: Direct SNS publish
echo "Test 1: Sending direct SNS notification..."
aws sns publish \
    --topic-arn "$SNS_TOPIC_ARN" \
    --subject "LLM Safety Study - Test Notification" \
    --message "This is a test. If you received this, SNS is configured correctly. Timestamp: $(date)"

echo "Check your email for the test message."
echo ""

# Test 2: Show alarm states
echo "Current alarm states:"
aws cloudwatch describe-alarms \
    --alarm-name-prefix llm-safety-study \
    --query 'MetricAlarms[*].[AlarmName,StateValue]' \
    --output table

echo ""
echo "Test complete. You should receive an email within 1-2 minutes."
