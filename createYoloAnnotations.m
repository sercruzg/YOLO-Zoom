function createYoloAnnotations()

path = '<path-to-images>';
impos = [];
load(['EgoDailyDatabase.mat']);

im = imread(impos(1).im);
[height width ch] = size(im);

for i = 1 : length(impos)
    
    fprintf(['Image: ' num2str(i) '\n']);
    [numHands temp] = size(impos(i).boxes);
    fileID = fopen([path 'egoDailyDatabase/labels/' impos(i).im(25:end-4) '.txt'],'w');
    for j = 1 : numHands
        box = impos(i).boxes(j,:);
        x = (box(1) + box(3))/2;
        y = (box(2) + box(4))/2;
        w = box(3) - box(1);
        h = box(4) - box(2);
        
        x = x / width;
        y = y / height;
        w = w / width;
        h = h / height;
        left = impos(i).left(j);
        fprintf(fileID, '%d %f %f %f %f\n', left, x, y, w, h);
    end
   fclose(fileID); 
   
end
end