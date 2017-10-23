import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

import featureExtractor as fe

def plotBestNgrams(keepBestN=50):
    ngramsList, ngramDict, ngramLangCounts = fe.processDictionaryAsTrainingFeatures(
        force_recompute_best=True, force_recompute_dict=False)

    print(ngramsList[:keepBestN])

    ngramCounts = np.zeros((keepBestN, len(fe.languageNames)))
    for index, ngram in enumerate(ngramsList[:keepBestN]):
        ngramCounts[index] = ngramDict[ngram][0]
    ngramCounts = ngramCounts.transpose()

    data = []
    for label, language in fe.languageNames.items():
        trace = go.Bar(
            name=language,
            x=ngramsList[:keepBestN],
            y=ngramCounts[label] / ngramLangCounts.sum() * 100,
            opacity=0.75
        )

        data.append(trace)

    layout = go.Layout(barmode='stack',
                       title="Distribution of top {} selected N-grams".format(keepBestN),
                       legend=dict(x=0.9, y=1),
                       xaxis=dict(
                           title='Character',
                           titlefont=dict(color='#262626')
                       ),
                       yaxis=dict(
                           title='Proportion in dictionary (%)',
                           titlefont=dict(color='#262626')
                       )
                       )
    fig = go.Figure(data=data, layout=layout)

    plot_url = py.plot(fig, filename='overlaid histogram')

# plotBestNgrams()

def plotNgramLenghtVsAccuracy():
    accuracies = [0.7976, 0.7294, 0.6685, 0.6126, 0.521, 0.4842, 0.49, 0.472]
    lengths = [i + 1 for i in range(8)]

    trace = go.Scatter(
        x=lengths,
        y=accuracies,
        opacity=0.95
    )

    data = [trace]
    layout = go.Layout(
                       legend=dict(x=0.9, y=1),
                       xaxis=dict(
                           title='Max N-gram length l',
                           titlefont=dict(color='#262626', size=18),
                       ),
                       yaxis=dict(
                           title='Classification accuracy (%)',
                           titlefont=dict(color='#262626', size=18),
                       )
                       )
    fig = go.Figure(data=data, layout=layout)

    plot_url = py.plot(fig, filename='accuracy')

plotNgramLenghtVsAccuracy()
