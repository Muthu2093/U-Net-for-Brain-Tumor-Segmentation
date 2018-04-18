function  image = im_align1(im1)
min_ssd_red = 999999999;
shift_pos_red = 0;
min_ssd_green = 999999999;
shift_pos_green = 0;
imshow(im1);
red_old  = im1(:,:,1);
green_old = im1(:,:,2);
blue_old  = im1(:,:,3);
red  = im1(:,:,1);
green = im1(:,:,2);
blue   = im1(:,:,3);
figure;
imshow(cat(3,red));
counter = -15;
    for k = 1:30
        ssd = cal_ssd(green,blue);
        disp(ssd);
        if ssd < min_ssd_green
            min_ssd_green = ssd;
            disp(counter);
            shift_pos_green = counter;
        end
        green = circshift(green,counter);
        counter = counter + 1;
    end
    green_old = circshift(green_old,counter);
    new_image = cat(3,red_old,green_old,blue_old);
    figure;
    imshow(new_image);
image = shift_pos_green;
end
 
