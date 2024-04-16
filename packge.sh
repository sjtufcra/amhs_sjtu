python3 -m build
cp -r dist/* ../cflab_amhs/computing/dist/
git add .
git commit -m $1
git pull
git push 
echo "push content: success!"
sh ../cflab_amhs/git.sh "update package"
echo "update cflab_amhs package: success!"