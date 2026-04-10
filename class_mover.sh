for class in $(< /home/nmoran/Documents/python/fish_detector/classes.txt); do
    mkdir -p "/home/nmoran/Downloads/multi_val/$class"
    find . -type f -name "*$class*" -exec cp -n {} "/home/nmoran/Downloads/multi_val/$class/" \;
done
