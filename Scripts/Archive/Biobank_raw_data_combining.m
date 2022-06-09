clc
files = dir;
ukb3079_rfMRI_25_IDs_2_4 = cell(length(files)-2,1);
ukb3079_rfMRI_100_IDs_2_4 = cell(length(files)-2,1);

Directory = '/media/share/16.1/Data/Biobank/dr_files_all/IDs_2_4/'

for k = 3:length(files)
    k
    folder_number = files(k).name;
    addpath([Directory folder_number])
    
    load_name = [Directory folder_number '/fMRI/rfMRI_25.dr/dr_stage1.txt'];
    load(load_name)
    ukb3079_rfMRI_25_IDs_2_4{k-2} = dr_stage1;
    

    load_name = [Directory folder_number '/fMRI/rfMRI_100.dr/dr_stage1.txt'];
    load(load_name)
    ukb3079_rfMRI_100_IDs_2_4{k-2} = dr_stage1;
    
    rmpath([Directory folder_number])
    
end


clc
files = dir;

ukb3079_rfMRI_100_IDs_2_1 = cell(length(files)-2,1);

Directory = '/media/share/16.1/Data/Biobank/dr_files/IDs_2_1/'

for k = 3:length(files)
    k
    folder_number = files(k).name;
    addpath([Directory folder_number])
    

    load_name = [Directory folder_number '/fMRI/rfMRI_100.dr/dr_stage1.txt'];
    load(load_name)
    ukb3079_rfMRI_100_IDs_2_1{k-2} = dr_stage1;
    
    rmpath([Directory folder_number])
    
end


clc
tic
files = dir;
ukb_IDs_3 = cell(length(files)-2,1);

for k = 3:length(files)
    k
    folder_number = files(k).name;
    ukb_IDs_3{k-2} = folder_number;
    
    
end

%[sort(subject_IDs) Subjects_IDs_txt]
% There are 820 subjects in the workspace12b that aren't in the time series
% files so let's find the IDs of these 820 to remove them

%first_set = setdiff(subject_IDs, Subjects_IDs_txt);
%second_set = setdiff(Subjects_IDs_txt, subject_IDs);

[val,idxa, idxb]  = intersect(subject_IDs, Subjects_IDs_txt);

% 43799 IDs in both sets
% 1899 IDs only in the first set
% 1080 IDs only in the second set

% 45698 IDs in the behavioural dataset (42881 sometimes??) (46743 in the BETA workspace)
% 44878 IDs in the MRI dataset

%ukb43799_rfMRI_25 = ukb44878_rfMRI_25_all(idxb)
%ukb43799_rfMRI_100 = ukb44878_rfMRI_100_all(idxb)



ukb43799_rfMRI_100_concat = NaN(21477586,100);
d = 0;

for i = 1:43799
    i
    sub = size(ukb43799_rfMRI_100{i},1);
    d = d + sub;
    ukb43799_rfMRI_100_concat(d-sub+1:d,:) = ukb43799_rfMRI_100{i};
    
end

ukb43799_rfMRI_25_concat = NaN(21477586,25);
d = 0;

for i = 1:43799
    i
    sub = size(ukb43799_rfMRI_25{i},1);
    d = d + sub;
    ukb43799_rfMRI_25_concat(d-sub+1:d,:) = ukb43799_rfMRI_25{i};
    
end






 - 12b same as 12b imaging subject IDs
 - 21269692 and 23531629 are in eBETA_IDPs but not in eBETA
 - There are none in workspace12b that aren't also in 12eBETA_IDPs (21269692 is in workspace12b and 12eBETA_IDPs but not in 12e_BETA)
 - 1936 IDs in 12eBETA_IDPs that aren't in MRI data
 - there's only 69 subjects in the MRI data that aren't in the 12eBETA_IDPs workspace (71 that aren't in 12eBETA - i.e. both 21269692 and 23531629)

Therefore, use 12eBETA workspace!!


-----------------------------------------




clc
files = dir;
ukb3079_rfMRI_25_IDs_2_4 = cell(length(files)-2,1);
ukb3079_rfMRI_100_IDs_2_4 = cell(length(files)-2,1);

Directory = '/media/share/16.1/Data/Biobank/dr_files_all/IDs_2_4/'

for k = 3:length(files)
    k
    folder_number = files(k).name;
    addpath([Directory folder_number])
    
    load_name = [Directory folder_number '/fMRI/rfMRI_25.dr/dr_stage1.txt'];
    load(load_name)
    ukb3079_rfMRI_25_IDs_2_4{k-2} = dr_stage1;
    

    load_name = [Directory folder_number '/fMRI/rfMRI_100.dr/dr_stage1.txt'];
    load(load_name)
    ukb3079_rfMRI_100_IDs_2_4{k-2} = dr_stage1;
    
    rmpath([Directory folder_number])
    
end


clc
files = dir;

ukb3079_rfMRI_100_IDs_2_1 = cell(length(files)-2,1);

Directory = '/media/share/16.1/Data/Biobank/dr_files/IDs_2_1/'

for k = 3:length(files)
    k
    folder_number = files(k).name;
    addpath([Directory folder_number])
    

    load_name = [Directory folder_number '/fMRI/rfMRI_100.dr/dr_stage1.txt'];
    load(load_name)
    ukb3079_rfMRI_100_IDs_2_1{k-2} = dr_stage1;
    
    rmpath([Directory folder_number])
    
end


clc
tic
files = dir;
ukb_IDs_3 = cell(length(files)-2,1);

for k = 3:length(files)
    k
    folder_number = files(k).name;
    ukb_IDs_3{k-2} = folder_number;
    
    
end

%[sort(subject_IDs) Subjects_IDs_txt]
% There are 820 subjects in the workspace12b that aren't in the time series
% files so let's find the IDs of these 820 to remove them

%first_set = setdiff(subject_IDs, Subjects_IDs_txt);
%second_set = setdiff(Subjects_IDs_txt, subject_IDs);

[val,idxa, idxb]  = intersect(subject_IDs, Subjects_IDs_txt);

% 43799 IDs in both sets
% 1899 IDs only in the first set
% 1080 IDs only in the second set

% 45698 IDs in the behavioural dataset (42881 sometimes??) (46743 in the BETA workspace)
% 44878 IDs in the MRI dataset

%ukb43799_rfMRI_25 = ukb44878_rfMRI_25_all(idxb)
%ukb43799_rfMRI_100 = ukb44878_rfMRI_100_all(idxb)



ukb43799_rfMRI_100_concat = NaN(21477586,100);
d = 0;

for i = 1:43799
    i
    sub = size(ukb43799_rfMRI_100{i},1);
    d = d + sub;
    ukb43799_rfMRI_100_concat(d-sub+1:d,:) = ukb43799_rfMRI_100{i};
    
end

ukb43799_rfMRI_25_concat = NaN(21477586,25);
d = 0;

for i = 1:43799
    i
    sub = size(ukb43799_rfMRI_25{i},1);
    d = d + sub;
    ukb43799_rfMRI_25_concat(d-sub+1:d,:) = ukb43799_rfMRI_25{i};
    
end




%%
% 
%  - 12b same as 12b imaging subject IDs
%  - 21269692 and 23531629 are in eBETA_IDPs but not in eBETA
%  - There are none in workspace12b that aren't also in 12eBETA_IDPs (21269692 is in workspace12b and 12eBETA_IDPs but not in 12e_BETA)
%  - 1936 IDs in 12eBETA_IDPs that aren't in MRI data
%  - there's only 69 subjects in the MRI data that aren't in the 12eBETA_IDPs workspace (71 that aren't in 12eBETA - i.e. both 21269692 and 23531629)
% 
% Therefore, use 12eBETA workspace!!