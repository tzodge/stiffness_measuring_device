% resource: https://blogs.mathworks.com/steve/2006/06/02/cell-segmentation/
clc
clear

I = imread('us_image416.jpg');
% I_cropped = I(90:180, 250:325);
I_cropped = I;
% [x,y] = ginput(n)
% rect = getrect(I_cropped)

[J,rect2] = imcrop(I_cropped);
rect2 = uint16(rect2);

rect_coord = [rect2(2),rect2(2)+rect2(4), rect2(1),rect2(1)+rect2(3)];
I_cropped = I_cropped(rect_coord(1):rect_coord(2), rect_coord(3):rect_coord(4));
% imshow(I_cropped)

% se = strel('line',11,90);
% se = offsetstrel('ball',1,1);
% I_cropped = imerode(I_cropped,se);
% I_cropped = imdilate(I_cropped,se);


% I_cropped = uint8(imbinarize(I_cropped,0.5)*255); 
I_cropped = process_image(I_cropped);
imshow(I_cropped)


%%
% imcredit('Image courtesy of Dr. Ramiro Massol')


I_eq = adapthisteq(I_cropped);
imshow(I_eq)
%%
bw = im2bw(I_eq, graythresh(I_eq));
imshow(bw)


bw2 = imfill(bw,'holes');
bw3 = imopen(bw2, ones(5,5));
bw4 = bwareaopen(bw3, 40);
bw4_perim = bwperim(bw4);
overlay1 = imoverlay(I_eq, bw4_perim, [.3 1 .3]);
imshow(overlay1)



function [img_out] = process_image(input_img)
    img_inverted = max(max(input_img)).*ones(size(input_img),'uint8') - input_img;
    clip_top = 255;
    img_inverted(img_inverted>clip_top ) =  clip_top ;
    
    clip_bottom = 0;
    img_inverted(img_inverted<clip_bottom) =  clip_bottom;
    img_out = img_inverted;
    
end
% 
% mask_em = imextendedmax(I_eq, 30);
% imshow(mask_em)
% 
% 
% mask_em = imclose(mask_em, ones(5,5));
% mask_em = imfill(mask_em, 'holes');
% mask_em = bwareaopen(mask_em, 40);
% overlay2 = imoverlay(I_eq, bw4_perim | mask_em, [.3 1 .3]);
% 
% 
% 
% imshow(overlay2)