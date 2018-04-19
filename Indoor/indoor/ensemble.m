clear all
close all
list_all=dir('./our_cvprw_test3sub/*png');

folder_name='./our_cvprw_submitted/';

if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end

for i=1:5
    hazy=imread(['./our_cvprw_test3sub/',list_all(i).name]);
    hazy2=imread(['./our_cvprw_test2sub/',list_all(i).name]);

    
    img1=hazy/2+hazy2/2;

    imwrite([img1],['./our_cvprw_submitted/',num2str(i+30),'.png']);

    
end