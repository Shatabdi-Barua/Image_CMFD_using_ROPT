clc;
clear all;
close all;

[filename, user_canceled] = imgetfile;
im=imread(filename);
tic; %record time for calculate elapsed time
[row colm]=size(im);

resz =imresize(im, [512 512]);
ir = imresize(im,[8,8]); %image resize
figure, imshow(ir);

 ih = rgb2hsv(ir); %convert rgb to hsv
 figure, imshow(ih);
 tic;
 ih_i = ih(:,:,3); %3rd matix
 ih_k = ih(:,:,1);
 ih_l = ih(:,:,2);
%%%===================ROPT
H =[1 1;
     1 -1];
 
 K1 = kron(H,H);
 K2 = kron(K1,H);
 K3 = kron(K2,H);
 
 v = [1 2+i 3+i 4+i 4+i 3+i 2+i 1];
 mul = zeros(8,8);
 for i= 1:8
    for j = 1:8
        if(i==j)
            if K2(i,j)== -1
                go = true;
                j =1;
                while go
                    mul(i,j)= K2(i,j).*v(1,j);
                    j = j+1;
                    if (j>8)
                        go= false;
                    end
                end
                if go == false
                        break;
                end
            else
                mul(i,j)= K2(i,j);
            end
        else
            mul(i,j) = K2(i,j);
        end
    end
 end

 in = inv(mul);
 
 
%%%%=============Multiply
 trMat = in.*ih_i;
 
trMat_real = abs(trMat); %convert to real number
 large = (max(trMat_real(:)));
%  largest = max(large); %max value
 divi = trMat_real./large; %divide each element of first matrix
 s=double(ir);
 trDiv = trMat_real/500;
 retr=imresize(trDiv,[512 512]);
 trAdd = retr+double(resz);
 tw=trAdd/255;
 imshow(tw);
 
 ttt=imresize(tw, [512 512]);
 img = rgb2gray(im);
I = resz;
I1 = I(1:256,1:256,3);
I2 = I(257:512,1:256,3);
I3 = I(1:256,257:512,3);
I4 = I(257:512, 257:512,3);
% figure, imshow(I1); 
% figure, imshow(I2);
% figure, imshow(I3);
% figure, imshow(I4);
I = [I1 I2 I3 I4];
ct=1;
if (ct==1)
for rs=1:4
    mh1 = I(rs);
 for br=1:4
        mh2 = I(br);
     [pv1,pv2] = SSIM(im,mh1,mh2);
   
       pv=[pv1;pv2];
          
      %%====Mask======
    inlen1=uint64(pv1);
    inlen2=uint64(pv2);
    
     [rw,cl] =size(img);
    len1=length(inlen1);
    len2 = length(inlen2);


    
    sum1 =0;
for i=1:len1
   
    matx1{i}  = zeros(rw,cl);
    B1 = matx1{i};
    ar=inlen1(i);
    bc=inlen1(i+len1);

  cr = ar-23;
  dc = bc-23;  
  for w=cr:ar+23
      for z=dc:bc+23
          B1(w,z)=1;
      end
  end
%    if size(matx1{i})~= [rw cl]
%        matx1{i}=imresize(matx1{i}, [rw cl]);
%     end
%     elseif size(matx1{i})==[rw cl]
    matx1{i}=B1;
    s=matx1{i};
    sum1 = sum1+matx1{i};
  
    sum1(sum1>0)=1;
%     figure,imshow(s)
  
end
  [rs1 cs1]=size(sum1);
   tw=imresize(tw, [rs1 cs1]);
mmn1= sum1.*im2double(tw);
%          figure, imshow(sum1);

A=zeros(512, 512);
new_sum1=sum1;
sum1=imresize(sum1,[512 512]);
total_sum1 = sum(sum1(:));
for gee=1:512
    for rat =1:512
      new(rat,gee) =  sum1(rat,gee);
      
      if rat<512 && gee<512
     
      new1(rat,gee) = sum1(rat+1, gee+1);
      end
    end
    tot=sum(new(:));
    total=sum(new1(:));
       if tot>0
% %         total=sum1(rat+1:cs1, gee+1:rs1);
%          if tot==total
%             uu=new;
%           break 
%          end
        if tot>total
            uu=new;
            break;
        elseif tot<total
                uu=new1;
        end
        
       end
end
if total_sum1>50000
  for a=1:rs1
    for d=1:cs1
        
       if a>1 && a<270 || d>1 && d<220
              if sum1(a,d)==1
                 new_sum1(a,d)=0;
              end
       end
 end
  end
end
% uu=imresize(uu,[row colm]);
% tw=imresize(tw,[row colm]);


sum2=0;
for j=1:len2
    matx2{j}  = zeros(rw,cl);
    B2 = matx2{j};
    ar=inlen2(j);
    bc=inlen2(j+len2);

    cr=ar-23;
    dc=bc-23;
    for o=cr:ar+23
        for p=dc:bc+23
            B2(o,p)=1;
        end
    end
    matx2{j}=B2;
    s2=matx2{j};
    sum2 = sum2+matx2{j};
     sum2(sum2>0)=1;
%     figure,imshow(s)
  
end
%           figure, imshow(sum2);
mmn2= sum2.*im2double(tw);
addt= mmn1+mmn2;
addt2=imresize(sum1,[512 512])+imresize(sum2, [512 512]);

fin = im2double(tw).*addt;

 end

   end
end 
imshow(addt); %musk with color image
imshow(addt2); %only musk 
% figure, imshow(fin);

    if sum1==0;
% figure,imshow(new_sum1);
    else
% figure, imshow(uu);
% figure, imshow(new_sum1);
    end
    
 orig=new_sum1.*im2double(tw);
% figure, imshow(orig);
% figure, imshow(new1);

[po, lo]=size(uu);
 for py=1:po
     for ly=1:lo
         A(py,ly)=uu(py,ly);
     end
 end
 A2= imresize(addt,[512 512])-A;
 
 
 ori=A.*im2double(resz);
 figure, imshow(A); %f2
 figure, imshow(ori); %f3
 figure, imshow(A2); %f4
 figure, imshow(orig); %f5
 A3 = 1-A2;
 figure, imshow(A3); %f6
 figure, imshow(new1); %f7
%  ssnn=snr(A3(:));
%  ssdd1 = std(A3(:));
 
toc;