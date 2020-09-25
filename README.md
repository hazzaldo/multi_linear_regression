# Data Science - Multi Linear Regression

**Project Scenario:**

The business problem description, that the Multiple Linear Regression is solving here, is the following:

We have the following dataset from 50 anonymous startups, where we collected their expenditure figures in the following areas:

- R&amp;D Spent
- Administration Spend
- Marketing Spend

We also have the location of the startup, i.e. in which US state. All the aforementioned 4 data columns are the input data (independent variables). Finally, we have the profit that each startup made in the last year. This will act as our output data (dependent variable) that the Multiple Linear Regression will try to predict on future data input data.

The imaginary challenge here is that a venture capitalist firm has hired a data scientist to analyse the dataset of the 50 startups and create a model that will tell the venture capitalist fund: which startups will give it the most return on its investment, by predicting profit given the aforementioned input data for the startup are provided.

Essentially, we&#39;re also trying to work out which area of expenditure (or a balance of expenditure between all 3 areas) will yield the best profit outcome. Also, we&#39;re trying to identify a pattern given the US stat that the startup is based in.

**To run this project:**

There are a number of ways to run the Python code for this project:

- **Options 1:** The easier and preferred way for me is simply to run the `multiple_linear_regression.ipynb` file in Google Collaboratory online tool: [https://colab.research.google.com/](https://colab.research.google.com/) . This tool will have everything setup for you - Python runtime, packages, tools ...etc. All you need to do is run your code.
- **Option 2:** run the `multiple_linear_regression.py` locally on your machine. For this option you need to ensure to have Python runtime, any packages imported/needed in the code as well as setup your virtual environment (preferred). With regards to version, the code should run on the latest version of Python runtime and packages.
- Ensure the dataset `50_Startups.csv` file is in the same directory as your Python code file, in order for the code to be able to import the dataset.

**Project outcome:**

Once you run the project code, the outcome should show an array of two columns. The first is the test set of the real profit figures, and the second is the predicted profit figures from the input test set.

**Background theory explanation:**

Assumptions:

The concept behind Multiple Linear Regression is that it&#39;s similar to Simple Linear Regression except we&#39;re trying to identify multiple causations that affects our outcome prediction:

![](img\img1.PNG)

Of course, linear regression has assumptions:

- Linearity -
- Homoscedasticity -
- Multivariate normality -
- Independence of errors -
- Lack of multicollinearity -

Therefore, before building a Linear Regression model, you need to check that these assumptions are all true.

Intuition:

Our profit is the data that we&#39;re trying to predict, therefore it&#39;s the dependent variable. While all the other columns are the input data (independent variables).

![](img\img2.PNG)

Therefore, our model equation will be the sum of all the independent variables (X), and where each independent variable is multiplied by a weight (b) that is adjusted by the model to optimise for the best prediction, plus a bias/coefficient (b0) to help adjust the outcome.

![](img\img3.PNG)

However, with regards to the US state column, this is a word and not a number value. State is a categorical variable. Therefore, we need to transform them into dummy variables. This process involves creating separate columns for each State category. Then marking a 1 or 0 on each row record, indicating whether the category State exists for the given row record.

![](img\img4.PNG)

Thus, when adding these dummy variables to our equation, we add an individual State column separately as an independent variable (as opposed to the original State column that contains all the states in string values). So D1 in this case will be New York column data. Also, bear in mind we ignore the other State columns as well (i.e. California and any other state). By including only one of the State columns is enough to preserve the data needed from the dummy variable column. Doing so will avoid the dummy variable trap. For categorical data it&#39;s more or less like having a switch, where the value will be either 1 or 0 across all categories. Hence these are not acting as separate linear input data, so we should not include them all or else the Linear Regression model will treat them as such, causing the dummy variable trap, ultimately causing our model to predict incorrectly.

![](img\img5.PNG)

Also, to bear in mind is that in real life scenario, you will be facing a lot of data. Some are relevant and significant to our model&#39;s prediction while others are simply noise. Hence you do not want to include all data in your model. Therefore, in building your Multiple Linear Regression model, there are 5 models (methods 2 to 4 are also referred to as Stepwise Regression):

1. All-in
2. Backward Elimination
3. Forward Selection
4. Bidirectional Elimination
5. Score Comparison

Here&#39;s a brief explanation about each model:

**All-in:**

When you use the input variables. This scenario applies usually when someone has already identified the relevant significant input dataset, and provides them ready for you to use them in your model. Or if you&#39;re preparing for the Backward Elimination method.

**Backward Elimination:** best explained as a step-by-step

1. Select a significant (P-value) threshold level to stay in the model (e.g. SL - 0.05).
2. Fit the full model with all possible predictors - i.e. use all your input variables in the model.
3. Consider the predictor with the highest P-value. If P \&gt; SL, go to step 4, otherwise go to the final step.
4. Remove the predictor.
5. Fit the model without this variable. Once you remove the variable, you have to re-fit the model.
6. Repeat step 3 to step 5 until you reach a point where all the remaining predictors&#39; P-value are less than the SL (significant level). As a result, your model is ready.

**Forward Selection:**

1. Select a significant level to enter the model (e.g. SL = 0.05).
2. Fit all input independent variables individually as simple regression models **Y ~ Xn**. Then select the one with the lowest P-value.
3. Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have. I.e. we keep the selected variable and we re-fit the model with one extra variable. We refit the selected variable with one extra variable, constructing models out of all the other variables added to the selected variable.
4. Consider the predictor with the lowest P-value. If P \&lt; SL, i.e. it&#39;s a significant variable, then go back to step 3 otherwise go to the final step. This means that we keep adding one more variable and refit the model, until the P \&lt; SL. Meaning all the remaining variables that we tried to add to the model are no longer significant variables.
5. Keep the previous model and not the current one that ended up with P \&gt; SL.

**Bidirectional Elimination:** this is of the most tedious methods and definitely requires a programmed algorithm to do this for you:

1. Select a significant level to enter and to stay in the model e.g.: SL\_ENTER = 0.05, and SL\_STAY = 0.05.
2. Perform the next step of Forward Selection (new variables must have: P \&lt; SLENTER to enter).
3. Perform ALL steps of Backward Elimination (old variables must have P \&lt; SLSTAY to stay).
4. Repeat step 2 - 3 until there are no new variables can enter and no old variables can exit. At this point your model is ready.

**All Possible Models:** this is the most thorough approach, but also the most resource consuming approach:

1. Select a criterion of goodness of fit (e.g. Akaike criterion).
2. Construct all possible Regression models: total combinations, where N is the number of independent variables.
3. Select the one with the best criterion. At this point your model is ready.

Example: if there are 10 columns in your data set - this results in 1,023 models. The 10 columns example is a raw dataset (i.e. not variables that you have filtered, preprocessed … etc.).

The method selected for this project:

We selected the Backward Elimination method because it is the fastest out of all the other methods.