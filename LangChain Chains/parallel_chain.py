from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

model1 = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
model2 = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
parser = StrOutputParser()
template1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)
template2 = PromptTemplate(
    template = "Generate 5 Quiz questions from the provided text. \n {text}",
    input_variables=['text']
)
template3 = PromptTemplate(
    template = "Merge Both these Notes and Quiz in a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)
parallel_chain = RunnableParallel (
    {'notes': template1 | model1 | parser,
    'quiz': template2 | model2 | parser}
)
merged_chain = template3 | model1 | parser
final_chain = parallel_chain | merged_chain
text = """For other uses, see Linear regression (disambiguation).
Part of a series on
Regression analysis
Models
Linear regressionSimple regressionPolynomial regressionGeneral linear model
Generalized linear modelVector generalized linear modelDiscrete choiceBinomial regressionBinary regressionLogistic regressionMultinomial logistic regressionMixed logitProbitMultinomial probitOrdered logitOrdered probitPoisson
Multilevel modelFixed effectsRandom effectsLinear mixed-effects modelNonlinear mixed-effects model
Nonlinear regressionNonparametricSemiparametricRobustQuantileIsotonicPrincipal componentsLeast angleLocalSegmented
Errors-in-variables
Estimation
Least squaresLinearNon-linear
OrdinaryWeightedGeneralizedGeneralized estimating equation
PartialTotalNon-negativeRidge regressionRegularized
Least absolute deviationsIteratively reweightedBayesianBayesian multivariateLeast-squares spectral analysis
Background
Regression validationMean and predicted responseErrors and residualsGoodness of fitStudentized residualGauss–Markov theorem
icon Mathematics portal
vte
In statistics, linear regression is a model that estimates the relationship between a scalar response (dependent variable) and one or more explanatory variables (regressor or independent variable). A model with exactly one explanatory variable is a simple linear regression; a model with two or more explanatory variables is a multiple linear regression.[1] This term is distinct from multivariate linear regression, which predicts multiple correlated dependent variables rather than a single dependent variable.[2]

In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. Most commonly, the conditional mean of the response given the values of the explanatory variables (or predictors) is assumed to be an affine function of those values; less commonly, the conditional median or some other quantile is used. Like all forms of regression analysis, linear regression focuses on the conditional probability distribution of the response given the values of the predictors, rather than on the joint probability distribution of all of these variables, which is the domain of multivariate analysis.

Linear regression is also a type of machine learning algorithm, more specifically a supervised algorithm, that learns from the labelled datasets and maps the data points to the most optimized linear functions that can be used for prediction on new datasets.[3]

Linear regression was the first type of regression analysis to be studied rigorously, and to be used extensively in practical applications.[4] This is because models which depend linearly on their unknown parameters are easier to fit than models which are non-linearly related to their parameters and because the statistical properties of the resulting estimators are easier to determine.

Linear regression has many practical uses. Most applications fall into one of the following two broad categories:

If the goal is error i.e. variance reduction in prediction or forecasting, linear regression can be used to fit a predictive model to an observed data set of values of the response and explanatory variables. After developing such a model, if additional values of the explanatory variables are collected without an accompanying response value, the fitted model can be used to make a prediction of the response.
If the goal is to explain variation in the response variable that can be attributed to variation in the explanatory variables, linear regression analysis can be applied to quantify the strength of the relationship between the response and the explanatory variables, and in particular to determine whether some explanatory variables may have no linear relationship with the response at all, or to identify which subsets of explanatory variables may contain redundant information about the response.
Linear regression models are often fitted using the least squares approach, but they may also be fitted in other ways, such as by minimizing the "lack of fit" in some other norm (as with least absolute deviations regression), or by minimizing a penalized version of the least squares cost function as in ridge regression (L2-norm penalty) and lasso (L1-norm penalty). Use of the Mean Squared Error (MSE) as the cost on a dataset that has many large outliers, can result in a model that fits the outliers more than the true data due to the higher importance assigned by MSE to large errors. So, cost functions that are robust to outliers should be used if the dataset has many large outliers. Conversely, the least squares approach can be used to fit models that are not linear models. Thus, although the terms "least squares" and "linear model" are closely linked, they are not synonymous"""
result = final_chain.invoke({'text': text})
print(result)
