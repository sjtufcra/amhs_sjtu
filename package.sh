python3 -m build
cp -r dist/* ../cflab_amhs/computing/dist/
git add .
git commit -am $1
git pull
git push 
echo "push content: success!"

cd ../cflab_amhs/
git add .
git commit -am $1
git pull
git push