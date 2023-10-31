## Approach 1
Do preprocessing on raw images
1. Bring RGB and MS to same resolution
2. Clip extra region of RGB to MS size
3. Copy all GPS and xmp data from RGB image to the resized MS images
4. Now construct orthomosaic

## Approach 2
Do post processing on orthomosaics
1. Get MS and RGB orthomosaic
2. Split with larger MS size and smaller RGB size
3. Keep significant overlap
4. Shift each image with alignment
5. MS will adjust because larger in size

## Apporach 3
Use Metashape for alignment
1. Import photos from each band into separate chunks
2. Align Photos in each chunk
3. Align all chunks
