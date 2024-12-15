import graphviz
dot = graphviz.Digraph()
dot.node('A', 'Hello')
dot.node('B', 'World')
dot.edges(['AB'])
dot.render('test_graphviz', format='png', view=True)