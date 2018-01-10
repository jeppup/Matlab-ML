function data = customreader(filename)
    I = imread(filename);
    data = imbinarize(I);
end
