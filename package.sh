python -m build
cp /s /i dist\* ..\cflab_amhs\computing\dist\
git add .
set /p commit_message="Enter commit message: "
git commit -m "%commit_message%"
git pull
git push
echo "Push content: success!"