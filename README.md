# Detecting Semantic Code Clones by Building AST-based Markov Chains Model
Amain is a scalable tree-based semantic code clone detector by building Markov chains models. We regard the nodes of an AST as different states and build a Markov chains model to transform the complex tree into simple state transitions. After collecting the distance vectors of all states between ASTs, we train a classifier that can assign suitable weights for different states to achieve succinct and effective semantic code clone detection.

Amain is mainly comprised of four phases: AST Generation, State Matrix Construction, Feature Extraction, and Classification.

1. AST Generation: The purpose of this phase is to apply static analysis to generate the corresponding AST. 
  The input of this phase is a method and the output is an AST.
2. State Matrix Construction: The purpose of this phase is to transform the AST into a Markov chain-based state matrix. The input of this phase is an AST and the output is a state transfer matrix. 
 3. Feature Extraction: The purpose of this phase is to calculate the distance vectors (\ie features) of two state transfer matrices. The input of this phase is two state transfer matrices and the output is a feature vector.
3. Classification: The purpose of this phase is to determine whether two methods are semantically similar or not. The input of this phase is a feature vector and the output is to report the detection results. 

The source code and dataset of Amain are published here.

# Project Structure  
  
```shell  
Amain  
|-- get_matrix.py     	// implement the first two phases:  AST Generation and State Matrix Construction
|-- get_distance.py     // implement the Feature Extraction phase  
|-- classification.py   // implement the Classification phase  
```

### get_matrix.py
- Input: dataset with source codes
- Output: state transfer matrices of source codes 
```
python get_matrix.py
```

### get_distance.py
- Input: state transfer matrices and code pairs
- Output: feature vectors of code pairs 
```
python get_distance.py
```

### classification.py
- Input: feature vectors of dataset
- Output: recall, precision, and F1 scores of machine learning algorithms
```
python classification.py
```

# 57 code syntax types
|Tool            |Type name                      |
|----------------|-------------------------------|
|0	|	MethodDeclaration|
|1	|	Modifier|
|2	|	FormalParameter|
|3	|	ReferenceType|
|4	|	BasicType|
|5	|	LocalVariableDeclaration|
|6	|	VariableDeclarator|
|7	|	MemberReference|
|8	|	ArraySelector|
|9	|	Literal|
|10	|	BinaryOperation|
|11	|	TernaryExpression|
|12	|	IfStatement|
|13	|	BlockStatement|
|14	|	StatementExpression|
|15	|	Assignment|
|16	|	MethodInvocation|
|17	|	Cast|
|18	|	ForStatement|
|19	|	ForControl|
|20	|	VariableDeclaration|
|21	|	TryStatement|
|22	|	ClassCreator|
|23	|	CatchClause|
|24	|	CatchClauseParameter|
|25	|	ThrowStatement|
|26	|	WhileStatement|
|27	|	ArrayInitializer|
|28	|	ReturnStatement|
|29	|	Annotation|
|30	|	SwitchStatement|
|31	|	SwitchStatementCase|
|32	|	ArrayCreator|
|33	|	This|
|34	|	ConstructorDeclaration|
|35	|	TypeArgument|
|36	|	EnhancedForControl|
|37	|	SuperMethodInvocation|
|38	|	SynchronizedStatement|
|39	|	DoStatement|
|40	|	InnerClassCreator|
|41	|	ExplicitConstructorInvocation|
|42	|	BreakStatement|
|43	|	ClassReference|
|44	|	SuperConstructorInvocation|
|45	|	ElementValuePair|
|46	|	AssertStatement|
|47	|	ElementArrayValue|
|48	|	TypeParameter|
|49	|	FieldDeclaration|
|50	|	SuperMemberReference|
|51	|	ContinueStatement|
|52	|	ClassDeclaration|
|53	|	TryResource|
|54	|	MethodReference|
|55	|	LambdaExpression|
|56	|	InferredFormalParameter|


# Parameter details of our comparative tools
|Tool            |Parameters                     |
|----------------|-------------------------------|
|SourcererCC	|Min lines: 6, Similarity threshold: 0.7            |
|Deckard      |Min tokens: 100, Stride: 2, Similarity threshold: 0.9 |
|RtvNN       |RtNN phase: hidden layer size: 400, epoch: 25, $\lambda_1$ for L2 regularization: 0.005, Initial learning rate: 0.003, Clipping gradient range: (-5.0, 5.0), RvNN phase: hidden layer size: (400, 400)-400, epoch: 5, Initial learning rate: 0.005, $\lambda_1$ for L2 regularization: 0.005, Distance threshold: 2.56    |
|ASTNN      |symbols embedding size: 128, hidden dimension: 100, mini-batch: 64, epoch: 5, threshold: 0.5, learning rate of AdaMax: 0.002  |
|SCDetector      |distance measure: Cosine distance, dimension of token vector: 100, threshold: 0.5, learning rate: 0.0001 |
|DeepSim      |Layers size: 88-6, (128x6-256-64)-128-32, epoch: 4, Initial learning rate: 0.001, $\lambda$ for L2 regularization: 0.00003, Dropout: 0.75 |
|CDLH      |Code length 32 for learned binary hash codes, size of word embeddings: 100 |
|TBCNN      |Convolutional layer dim size: 300，dropout rate: 0.5, batch size: 10 |
|FCCA      |Size of hidden states: 128(Text), 128(AST), embedding size: 300(Text), 300(AST), 64(CFG) clipping gradient range: (-1.2，1.2), epoch: 50, initial learning rate: 0.0005, dropout:0.6, batchsize: 32|


