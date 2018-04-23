%%[V,info]=ReadData3D("/Users/muthuvel/Desktop/Independent Study - M Gao/BRATS2013/BRATS_Training/Image_Data/HG/0001/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.684.mha");
data="/Users/muthuvel/Desktop/Independent Study - M Gao/Dataset/BRATS_2013_Training/Image_Data/LG/0014/VSD.Brain_2more.XX.XX.OT/VSD.Brain_2more.XX.XX.OT.6616.mha";
files = dir('/Users/muthuvel/Desktop/Independent Study - M Gao/Dataset/BRATS_2013_Training/Image_Data/LG');





header=mha_read_header("test.mha");
volume=mha_read_volume(header);
figure(1)
subplot(3,2,1)
imshow(squeeze(volume(:,:,round(end/2))),[]);

header1=mha_read_header("test1.mha");
volume1=mha_read_volume(header1);
subplot(3,2,2)
imshow(squeeze(volume1(:,:,round(end/2))),[]);

header2=mha_read_header("test2.mha");
volume2=mha_read_volume(header2);
subplot(3,2,3)
imshow(squeeze(volume2(:,:,round(end/2))),[]);

header3=mha_read_header("test3.mha");
volume3=mha_read_volume(header3);
subplot(3,2,4)
imshow(squeeze(volume3(:,:,round(end/2))),[]);

header4=mha_read_header("test4.mha");
volume4=mha_read_volume(header4);
subplot(3,2,5)
imshow(squeeze(volume4(:,:,round(end/2))),[]);

figure(2)
vol=volume+volume1+volume2+volume3+volume4;
imshow(squeeze(vol(:,:,round(end/2))),[]);

figure(3)
vol=cat(3,volume,volume1,volume2,volume3,volume4);
imshow(squeeze(vol(:,:,round(end/2))),[]);
