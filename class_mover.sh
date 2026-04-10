for class in $(< ~/fish_classes.txt); do
    mkdir "/Users/nmoran/Downloads/train/$class"
    find . -type f -name "*$class*" -exec cp -n {} "/Users/nmoran/Downloads/train/$class/" \;
done
