% reference: https://www.mathworks.com/help/images/detecting-a-cell-using-image-segmentation.html

clc
clear

for i = 1
    data_file = 'data_2Sep_2';
    fileList = dir(strcat(data_file,'/*.jpg'));

    I = imread(strcat(data_file,'/',fileList(i).name));
    % I_cropped = I(90:180, 250:325);
    I_cropped = I;
    % [x,y] = ginput(n)
    % rect = getrect(I_cropped)

    [J,rect2] = imcrop(I_cropped);
    rect2 = uint16(rect2);

    rect_coord = [rect2(2),rect2(2)+rect2(4), rect2(1),rect2(1)+rect2(3)];
    I_cropped = I_cropped(rect_coord(1):rect_coord(2), rect_coord(3):rect_coord(4));


    [~,threshold] = edge(I_cropped,'sobel');
    fudgeFactor = 0.5;
    BWs = edge(I_cropped,'sobel',threshold * fudgeFactor);

    imshow(BWs)
    title('Binary Gradient Mask')

    se90 = strel('line',3,90);
    se0 = strel('line',3,0);
    BWsdil = imdilate(BWs,[se90 se0]);
    % imshow(BWsdil)
    % title('Dilated Gradient Mask')


    BWdfill = imfill(BWsdil,'holes');

    BWdfill  = imdilate(BWdfill ,[se90 se0]);
    imshow(BWdfill)
    title('Binary Image with Filled Holes')

    uint8Image = uint8(255 * (1-BWdfill));
    I_eq = adapthisteq(uint8Image );
    imshow(I_eq)

    bw = im2bw(I_eq, graythresh(I_eq));
    imshow(bw)

    bw2 = imfill(bw,'holes');
    bw3 = imopen(bw2, ones(5,5));
    bw4 = bwareaopen(bw3, 40);
    bw4_perim = bwperim(bw4);
    overlay1 = imoverlay(I_eq, bw4_perim, [.3 1 .3]);
    imshow(overlay1)
    w = waitforbuttonpress;
end
