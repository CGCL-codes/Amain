import numpy as np
import javalang
from javalang.ast import Node
from anytree import AnyNode
import time
import os
from multiprocessing import Pool
from functools import partial


nodetypedict = {'MethodDeclaration': 0, 'Modifier': 1, 'FormalParameter': 2, 'ReferenceType': 3, 'BasicType': 4,
     'LocalVariableDeclaration': 5, 'VariableDeclarator': 6, 'MemberReference': 7, 'ArraySelector': 8, 'Literal': 9,
     'BinaryOperation': 10, 'TernaryExpression': 11, 'IfStatement': 12, 'BlockStatement': 13, 'StatementExpression': 14,
     'Assignment': 15, 'MethodInvocation': 16, 'Cast': 17, 'ForStatement': 18, 'ForControl': 19,
     'VariableDeclaration': 20, 'TryStatement': 21, 'ClassCreator': 22, 'CatchClause': 23, 'CatchClauseParameter': 24,
     'ThrowStatement': 25, 'WhileStatement': 26, 'ArrayInitializer': 27, 'ReturnStatement': 28, 'Annotation': 29,
     'SwitchStatement': 30, 'SwitchStatementCase': 31, 'ArrayCreator': 32, 'This': 33, 'ConstructorDeclaration': 34,
     'TypeArgument': 35, 'EnhancedForControl': 36, 'SuperMethodInvocation': 37, 'SynchronizedStatement': 38,
     'DoStatement': 39, 'InnerClassCreator': 40, 'ExplicitConstructorInvocation': 41, 'BreakStatement': 42,
     'ClassReference': 43, 'SuperConstructorInvocation': 44, 'ElementValuePair': 45, 'AssertStatement': 46,
     'ElementArrayValue': 47, 'TypeParameter': 48, 'FieldDeclaration': 49, 'SuperMemberReference': 50,
     'ContinueStatement': 51, 'ClassDeclaration': 52, 'TryResource': 53, 'MethodReference': 54,
     'LambdaExpression': 55, 'InferredFormalParameter': 56}
tokendict = {'DecimalInteger': 57, 'HexInteger': 58, 'Identifier': 59, 'Keyword': 60, 'Modifier': 61, 'Null': 62,
              'OctalInteger': 63, 'Operator': 64, 'Separator': 65, 'String': 66, 'Annotation': 67, 'BasicType': 68,
              'Boolean': 69, 'DecimalFloatingPoint': 70, 'HexFloatingPoint': 71}


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


def getast(path):
    programfile = open(path, encoding='utf-8')
    # print(os.path.join(rt,file))
    programtext = programfile.read()
    # programtext=programtext.replace('\r','')
    programfile.close()
    programtokens = javalang.tokenizer.tokenize(programtext)
    # print("programtokens", list(programtokens))
    parser = javalang.parse.Parser(programtokens)
    programast = parser.parse_member_declaration()

    # print(programast)
    return programast


def get_token(node):
    token = ''
    # print(isinstance(node, Node))
    # print(type(node))
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    # print(node.__class__.__name__,str(node))
    # print(node.__class__.__name__, node)
    return token


def get_child(root):
    # print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    # print(sub_item)
                    yield sub_item
            elif item:
                # print(item)
                yield item

    return list(expand(children))


def createtree(root, node, nodelist, parent=None):
    id = len(nodelist)
    # print(id)
    token, children = get_token(node), get_child(node)
    if id == 0:
        root.token = token
        root.data = node
    else:
        newnode = AnyNode(id=id, token=token, data=node, parent=parent)
    nodelist.append(node)
    for child in children:
        if id == 0:
            createtree(root, child, nodelist, parent=root)
        else:
            createtree(root, child, nodelist, parent=newnode)


def getnodeandedge(node, src, tgt):
    for child in node.children:
        src.append(node.token)
        tgt.append(child.token)
        getnodeandedge(child, src, tgt)


def one_matrix(path, pkl_path=None):
    # ast generation
    programfile = open(path, encoding='utf-8')
    # print(os.path.join(rt,file))
    programtext = programfile.read()
    # programtext=programtext.replace('\r','')
    programtokens = javalang.tokenizer.tokenize(programtext)
    parser = javalang.parse.Parser(programtokens)
    tree = parser.parse_member_declaration()
    programfile.close()

    file = open(path, "r", encoding='utf-8')
    tokens = list(javalang.tokenizer.tokenize(file.read()))
    #print("programtokens", list(tokens))
    file.close()

    # create tree
    nodelist = []
    newtree = AnyNode(id=0, token=None, data=None)
    createtree(newtree, tree, nodelist)

    if pkl_path != None:
        out_pkl = pkl_path + path.split('/')[-1].split('.java')[0] + '.pkl'
        WriteAndRead.write_pkl(out_pkl, newtree)

    # token type dictionary
    typedict = {}
    for token in tokens:
        token_type = str(type(token))[:-2].split(".")[-1]
        token_value = token.value
        if token_value not in typedict:
            typedict[token_value] = token_type
        else:
            if typedict[token_value] != token_type:
                print('!!!!!!!!')

    # matrix initialization
    matrix = [[0 for col in range(72)] for row in range(57)]

    # Traverse the tree to get edge information
    src = []
    tgt = []
    getnodeandedge(newtree, src, tgt)

    #print(len(src))

    # fill matrix
    for i in range(len(src)):
        m = nodetypedict[src[i]]
        name = tgt[i]
        try:
            n = nodetypedict[name]
        except KeyError:
            try:
                n = tokendict[typedict[name]]
            except KeyError:
                n = 62
        matrix[m][n] += 1

    for k in range(57):
        total = 0
        for l in range(72):
            total += matrix[k][l]
        if total != 0:
            for p in range(72):
                matrix[k][p] = matrix[k][p]/total

    matrix = np.array(matrix)
    filename = path.split('/')[-1].split('.java')[0]
    print(filename)
    npypath = './npy/' + filename
    np.save(npypath, matrix)
    return matrix


javapath = './dataset/googlejam4/'
def allmain():
    # Read all java files from a folder
    javalist = []
    listdir(javapath, javalist)

    for javafile in javalist:
        try:
            one_matrix(javafile)
        except (UnicodeDecodeError, javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError):
            print(javafile)


if __name__ == '__main__':
    allmain()

