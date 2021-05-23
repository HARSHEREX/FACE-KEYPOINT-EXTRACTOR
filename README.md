# FACE-KEYPOINT-EXTRACTOR
a complex tool to solve facial project challenges and useful for anyone who want to get face key points for any king of facial project (face recognition / expression recognition)

## user must train the model again because model size is 200MB + so it can not be uploaded to github
	* program will ask you to retrain model 

# model performance :
	orange is validation and blue is training
 - ![alt text](
https://github.com/HARSHEREX/FACE-KEYPOINT-EXTRACTOR/blob/main/Resource/model/performance%20graph.jpg?raw=true)

# Example :
 -  query images (any resolution but should be 1:1 aspect ratio or 1(±5%) :1(±5%)  for best results

	 - ![alt text](https://github.com/HARSHEREX/FACE-KEYPOINT-EXTRACTOR/blob/main/Resource/sample%20query%20and%20results/harshit.01.20.jpg?raw=true)
- Result is 96*96 px

	 -  ![alt text](https://github.com/HARSHEREX/FACE-KEYPOINT-EXTRACTOR/blob/main/Resource/sample%20query%20and%20results/harshit.01.20.jpg_result_.jpg?raw=true)
	
*	Result file for above sample  it contails 31 columns 0-30 are location coordinates for facial points and last one is name of query image

| left\_eye\_center\_x | left\_eye\_center\_y | right\_eye\_center\_x | right\_eye\_center\_y | left\_eye\_inner\_corner\_x | left\_eye\_inner\_corner\_y | left\_eye\_outer\_corner\_x | left\_eye\_outer\_corner\_y | right\_eye\_inner\_corner\_x | right\_eye\_inner\_corner\_y | right\_eye\_outer\_corner\_x | right\_eye\_outer\_corner\_y | left\_eyebrow\_inner\_end\_x | left\_eyebrow\_inner\_end\_y | left\_eyebrow\_outer\_end\_x | left\_eyebrow\_outer\_end\_y | right\_eyebrow\_inner\_end\_x | right\_eyebrow\_inner\_end\_y | right\_eyebrow\_outer\_end\_x | right\_eyebrow\_outer\_end\_y | nose\_tip\_x | nose\_tip\_y | mouth\_left\_corner\_x | mouth\_left\_corner\_y | mouth\_right\_corner\_x | mouth\_right\_corner\_y | mouth\_center\_top\_lip\_x | mouth\_center\_top\_lip\_y | mouth\_center\_bottom\_lip\_x | mouth\_center\_bottom\_lip\_y | name              |
| -------------------- | -------------------- | --------------------- | --------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ------------ | ------------ | ---------------------- | ---------------------- | ----------------------- | ----------------------- | -------------------------- | -------------------------- | ----------------------------- | ----------------------------- | ----------------- |
| 64.91113             | 41.35905             | 30.76946              | 40.28876              | 57.41777                    | 41.31385                    | 72.86856                    | 41.40854                    | 37.13642                     | 41.83436                     | 22.97009                     | 41.56949                     | 55.70026                     | 31.73107                     | 77.89137                     | 31.97036                     | 39.8092                       | 32.39046                      | 17.29502                      | 32.93769                      | 47.00873     | 56.88908     | 64.62939               | 76.58209               | 32.50082                | 76.53281                | 48.07755                   | 75.42761                   | 48.57299                      | 82.41991                      | harshit.01.20.jpg |
