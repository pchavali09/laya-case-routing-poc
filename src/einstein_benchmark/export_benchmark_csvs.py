import os
import json
import random
import pandas as pd

SEED = 2026
random.seed(SEED)

all_depts = ['billing', 'technical', 'sales', 'operations', 'other']
all_cases = []
for d in all_depts:
    with open(f'src/einstein_benchmark/data_{d}.json') as f:
        all_cases.extend(json.load(f))

train_cases = [c for c in all_cases if c['split'] == 'train']
test_cases = [c for c in all_cases if c['split'] == 'test']

assert len(train_cases) == 800, f"Expected 800 train cases, got {len(train_cases)}"
assert len(test_cases) == 200, f"Expected 200 test cases, got {len(test_cases)}"

# Shuffle deterministically
random.shuffle(train_cases)
random.shuffle(test_cases)

# Assign External IDs
for i, c in enumerate(train_cases, 1):
    c['ExternalId__c'] = f"ENS-TRAIN-{i:04d}"

for i, c in enumerate(test_cases, 1):
    c['ExternalId__c'] = f"ENS-TEST-{i:04d}"

out_dir = "data/einstein_benchmark"
repo_dir = os.path.join(out_dir, "repo_format")
os.makedirs(repo_dir, exist_ok=True)

# 1. train_closed.csv
train_closed_df = pd.DataFrame([{
    "ExternalId__c": c["ExternalId__c"],
    "Subject": c["subject"],
    "Description": c["description"],
    "Origin": c["origin"],
    "Priority": c["priority"],
    "Status": "Closed",
    "Department__c": c["department"]
} for c in train_cases])
train_closed_df.to_csv(os.path.join(out_dir, "train_closed.csv"), index=False)

# 2. test_blind.csv
test_blind_df = pd.DataFrame([{
    "ExternalId__c": c["ExternalId__c"],
    "Subject": c["subject"],
    "Description": c["description"],
    "Origin": c["origin"],
    "Priority": c["priority"],
    "Status": "New",
    "Department__c": ""
} for c in test_cases])
test_blind_df.to_csv(os.path.join(out_dir, "test_blind.csv"), index=False)

# 3. test_truth.csv
test_truth_df = pd.DataFrame([{
    "ExternalId__c": c["ExternalId__c"],
    "true_department": c["department"],
    "difficulty": c["difficulty"]
} for c in test_cases])
test_truth_df.to_csv(os.path.join(out_dir, "test_truth.csv"), index=False)

# 4. repo_format CSVs
def to_repo_row(c, is_test=False):
    # urgency maps from priority (Low, Medium, High -> High maps to Critical or High)
    urgency = "Critical" if c["priority"] == "High" else c["priority"]
    # churn_risk is Yes if angry and high priority, else No
    churn = "Yes" if (c["voice"] == "angry" or c["priority"] == "High") else "No"
    return {
        "case_id": c["ExternalId__c"],
        "subject": c["subject"],
        "body": c["description"],
        "department": c["department"],
        "urgency": urgency,
        "churn_risk": churn,
        "language": "English"
    }

repo_train_df = pd.DataFrame([to_repo_row(c) for c in train_cases])
repo_train_df.to_csv(os.path.join(repo_dir, "train.csv"), index=False)

repo_test_df = pd.DataFrame([to_repo_row(c, is_test=True) for c in test_cases])
repo_test_df.to_csv(os.path.join(repo_dir, "test.csv"), index=False)

repo_all_df = pd.concat([repo_train_df, repo_test_df], ignore_index=True)
repo_all_df.to_csv(os.path.join(repo_dir, "all_cases.csv"), index=False)

print("Export completed successfully!")
print(f"Files written in {out_dir}:")
for f in os.listdir(out_dir):
    p = os.path.join(out_dir, f)
    if os.path.isfile(p):
        print(f"  {f} ({os.path.getsize(p)} bytes)")
print(f"Files written in {repo_dir}:")
for f in os.listdir(repo_dir):
    p = os.path.join(repo_dir, f)
    if os.path.isfile(p):
        print(f"  {f} ({os.path.getsize(p)} bytes)")
