clear all
close all

list_all=dir('./our_cvprw_test/*png');
list_all1=dir('./gtval/*png');
list_all2=dir('./our_cvprw_test2/*png');

% size_all=[2833,4657;2833,4657;2802,4516;2812,4356;2472,3936];
size_all=[2942,2426;3412,4056;3352,4846;3482,2636;2602,3696];


folder_name='./our_cvprwoutdoor_submitted/';
if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end

for i=1:5
    img1=imread(['./our_cvprw_test/',list_all(i).name]);
%     img1=imresize(img1,size_all(i,:));

%     input=imread(['./input_test/',list_all2(i).name]);
    hazy2=imread(['./our_cvprw_test2/',list_all2(i).name]);

        
    
%     gt=imread(['./gtval/',list_all1(i).name]);
    
        img1=imresize(img1,size_all(i,:));
        hazy2=imresize(hazy2,size_all(i,:));

%     img1=imhistmatch(img1,input);

    img3=img1/2+hazy2/2;


%     psnr_all3(i)=psnr(img3,gt);
%     psnr_all1(i)=psnr(input,gt);

    imwrite(img3, ['./our_cvprwoutdoor_submitted/',num2str(i+40),'.png']);


%     mean(psnr_all3)
%     mean(psnr_all1)

end
