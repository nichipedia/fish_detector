for class in $(< /home/nmoran/Documents/python/fish_detector/classes.txt); do
    mkdir -p "/home/nmoran/Downloads/multi_test/$class"
    find . -type f -name "*$class*" -exec cp -n {} "/home/nmoran/Downloads/multi_test/$class/" \;
done
