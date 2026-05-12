import sys
import os

print(f"Python Version: {sys.version}")
print("Checking packages...")

packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'xgboost', 'imblearn', 'scipy', 'jupyter', 'openpyxl']
all_passed = True

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ PASS: {pkg}")
    except ImportError:
        print(f"❌ FAIL: {pkg} is missing")
        all_passed = False

ibm_path = "data/raw/ibm_hr_attrition.csv"
li_path = "data/raw/linkedin_jobs.csv"

if os.path.exists(ibm_path):
    print("✅ PASS: IBM HR Dataset found")
else:
    print(f"❌ FAIL: IBM HR Dataset missing at {ibm_path}")
    all_passed = False

if os.path.exists(li_path):
    print("✅ PASS: LinkedIn Dataset found")
else:
    print(f"❌ FAIL: LinkedIn Dataset missing at {li_path}")
    all_passed = False

if all_passed:
    print("\n🎉 ALL CHECKS PASSED. You are ready to run the notebooks.")
else:
    print("\n⚠️ Checks failed. Fix the items above before running notebooks.")
