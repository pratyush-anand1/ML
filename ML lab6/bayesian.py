from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the structure of the Bayesian Network
model = BayesianNetwork([('e', 'm'), ('i', 's'), ('i', 'm'), ('m', 'a')])

# Step 2: Define Conditional Probability Distributions (CPDs)

# CPD for Exam level (e)
cpd_e = TabularCPD(variable='e', variable_card=2, values=[[0.7], [0.3]], state_names={'e': ['easy', 'difficult']})

# CPD for IQ level (i)
cpd_i = TabularCPD(variable='i', variable_card=2, values=[[0.8], [0.2]], state_names={'i': ['low', 'high']})

# CPD for Marks (m) given Exam level (e) and IQ level (i)
cpd_m = TabularCPD(variable='m', variable_card=2,
                   values=[[0.6, 0.9, 0.5, 0.8],   # P(m='low' | e='easy', i='low')
                           [0.4, 0.1, 0.5, 0.2]],  # P(m='low' | e='difficult', i='low')
                   evidence=['i', 'e'], evidence_card=[2, 2],
                   state_names={'m': ['low', 'high'], 'e': ['easy', 'difficult'], 'i': ['low', 'high']})

cpd_s = TabularCPD(variable='s', variable_card=2,
                   values=[[0.75, 0.4],  # P(s='low' | i='low')
                           [0.25, 0.6]],    # P(s='low' | i='high')
                   evidence=['i'], evidence_card=[2],
                   state_names={'s': ['low', 'high'], 'i': ['low', 'high']})



# CPD for Admission (a) given Marks (m)
cpd_a = TabularCPD(variable='a', variable_card=2,
                   values=[[0.6, 0.9],  # P(a | m='low')
                           [0.4, 0.1]],  # P(a | m='high')
                   evidence=['m'], evidence_card=[2],
                   state_names={'a': ['no', 'yes'], 'm': ['low', 'high']})

# Add CPDs to the model
model.add_cpds(cpd_e, cpd_i, cpd_m, cpd_s, cpd_a)

# Step 3: Check if the model is valid (all CPDs are consistent)
assert model.check_model()

# Step 4: Create an inference object
inference = VariableElimination(model)

# Step 5: Perform inference to get conditional probabilities of Marks (m) given Exam level (e)
result = inference.query(variables=['m'], evidence={'e': 'easy'})
result1 = inference.query(variables=['m'],evidence={'e':'difficult'})
print("Probability of Marks given Exam level is EASY")
print(result)
print("Probability of Marks given Exam level is HARD")
print(result1)