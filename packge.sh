python3 -m build
cp -r dist/* ../cflab_amhs/computing/dist/
sh ../cflab_amhs/git.sh "update package"