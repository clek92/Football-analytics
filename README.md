# Football-analytics
Football is one of the most famous sports and there are a lot of fans who try to predict football match outcomes <br>
This is a very simple project which uses a MLP model with 3 hidden layers in order to classify the number of goals scored in a football match <br>
The data were gathered with scraping techniques and include almost 4.200 matches of the English Premier League and they will remain non available for now <br>
The main idea is to classify matches according to the betting industry <br>
In betting industry over or under 2.5 goals is reffered as more than 2.5 total goals or less than 2.5 total goals scored by the opposing teams <br>
The model classifies the matches in the same way <br>
If there are more than 2.5 goals scored in a match then the target variable is set to 1 otherwise (meaning less than 2.5 goals) is set to 0 <br>
For the time being the repository will remain as is. <br>
Next steps: 
1) Reform the current file (it is one of my first projects) with scaleable code in order to be easily used
2) Gather more data reaching 10.000 football matches 
3) Conduct EDA for the new dataset
4) Construct new features for teams (currently only 13 available)
5) Update the model and improve its architecture
