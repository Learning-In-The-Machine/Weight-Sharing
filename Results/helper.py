import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from copy import deepcopy
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist, cifar10

SAVE_TYPE = 'png'
FIG_HEIGHT = 7

legend_settings = {'size': 16}
FONTSIZE = 30

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = x_train[7].squeeze()

'''
rm -rf Figures
mkdir Figures
mkdir Figures/Paper
mkdir Figures/SM
'''

titles = {
    'acc': 'Training Accuracy',
    'val_acc': 'Un-Augmented Validation Accuracy',
    'translation': 'Translation Augmented Validation Accuracy',
    'noise': 'Noise Augmented Validation Accuracy',
    'edge_noise': 'Edge Noise Augmented Validation Accuracy',
    'rotation': 'Rotation Augmented Validation Accuracy',
    'swap': 'Swap Quadrants Validation Accuracy',
}


def load_results(dataset='mnist', SET=True):
    all_results = pd.read_csv(dataset+'.csv')

    if SET:
        for aug_type in ['noise', 'edge_noise', 'rotation']:
            df = all_results[(all_results['aug_type'] == 'translation') & (all_results['aug'] == 0)].copy(deep=True)
            df['aug_type'] = aug_type

            all_results = all_results.append(df, ignore_index=True)

        for aug in [0, .1, .2, .3, .4]:
            df = all_results[(all_results['model'] == 'fcn') & (all_results['aug_type'] == 'translation') & (all_results['aug'] == aug)].copy(deep=True)
            df['aug_type'] = 'vcp'

            all_results = all_results.append(df, ignore_index=True)

    no_vcp_results = all_results[all_results['aug_type'] != 'vcp']

    cnn_results = no_vcp_results[no_vcp_results['model'] == 'cnn'].reset_index()
    fcn_results = no_vcp_results[no_vcp_results['model'] == 'fcn'].reset_index()
    fcn_results_vcp = all_results[all_results['aug_type'] == 'vcp'].reset_index()

    return no_vcp_results, cnn_results, fcn_results, fcn_results_vcp

def convert_table(df, to_latex=False):
    res_table = df.groupby(['model','aug_type','aug'])[['acc', 'val_acc', 'translation']].median().reset_index().pivot_table(index='aug',columns=['aug_type', 'model'])
    res_table.columns = res_table.columns.swaplevel(0,1)
    res_table.sort_index(axis=1, level=0, inplace=True)
    # res_table.columns = res_table.columns.droplevel()

    if to_latex: print res_table.to_latex()
    else: return res_table

def convert_table_vcp(df, eval_type='val_acc', to_latex=False):
    res_table = df.groupby(['vcp','aug'])[eval_type].median().to_frame().reset_index().pivot_table(index='vcp',columns=['aug'])
    res_table.columns = res_table.columns.droplevel()
    if to_latex: print res_table.to_latex()
    else: return res_table

def aug_eval(df, aug_type='translation', y=[0.9, 1.0], eval_type='val_acc'):
    hue = 'vcp' if aug_type == 'vcp' else 'aug'
    df_aug_type = df[df['aug_type'] == aug_type]

    palette = sns.color_palette('coolwarm', len(df_aug_type[hue].unique()))

    ax=sns.lineplot(x='epoch', y=eval_type, data=df_aug_type, hue=hue, palette=palette, legend='full')
    plt.ylim(y[0],y[1]); plt.xlabel('Epochs', fontsize=FONTSIZE); plt.ylabel('Accuracy', fontsize=FONTSIZE)

    ax.legend(ncol=4,bbox_to_anchor=(.5, .29), loc='upper center', shadow=True, prop=legend_settings).texts[0].set_text('VCP \%' if hue == 'vcp' else 'Aug \%')

def aug_train(cnn, fcn, aug_type='translation', y=[0.9, 1.0], eval_type='val_acc'):
    cnn_aug_type = cnn[cnn['aug_type'] == aug_type]
    fcn_aug_type = fcn[fcn['aug_type'] == aug_type]

    cnn_palette = sns.color_palette('coolwarm', len(cnn_aug_type['aug'].unique()))
    fcn_palette = sns.color_palette('coolwarm', len(fcn_aug_type['aug'].unique()))

    plt.subplot(1,2,2);
    sns.lineplot(x='epoch', y=eval_type, data=cnn_aug_type, hue='aug', palette=cnn_palette, legend='full')
    plt.ylim(y[0],y[1]); plt.xlabel('Epochs'); plt.ylabel(titles[eval_type])

    plt.subplot(1,2,1);
    sns.lineplot(x='epoch', y=eval_type, data=fcn_aug_type, hue='aug', palette=fcn_palette, legend='full')
    plt.ylim(y[0],y[1]); plt.xlabel('Epochs'); plt.ylabel(titles[eval_type]); plt.legend(shadow=True); plt.show()

def approx_ws(df, dist_type='cos', aug_type='translation',y=[], save=True, cifar=False):
    df = df[df['aug_type'] ==  aug_type].reset_index()
    palette = sns.color_palette('coolwarm', len(df['aug'].unique()))

    plt.clf(); fig=plt.figure(figsize=(20, FIG_HEIGHT))
    plt.subplot(1,2,1);
    ax=sns.lineplot(x='epoch', y=dist_type+'_1', data=df, hue='aug',palette=palette,legend='full')
    plt.ylim(y[0],y[1]); plt.xlabel('Epochs', fontsize=FONTSIZE); plt.title('(a) Filters at Distance 1', fontsize=FONTSIZE); plt.ylabel('Cosine Similarity' if dist_type=='cos' else 'Euclidean Distance', fontsize=FONTSIZE)
    ax.legend(ncol=4,bbox_to_anchor=(.5, .29), loc='upper center', shadow=True, prop=legend_settings).texts[0].set_text('Aug %')

    plt.subplot(1,2,2);
    ax=sns.lineplot(x='epoch', y=dist_type+'_4', data=df, hue='aug',palette=palette,legend='full')
    plt.ylim(y[0],y[1]); plt.xlabel('Epochs', fontsize=FONTSIZE); plt.title('(b) Filters at Distance 4', fontsize=FONTSIZE); plt.ylabel('Cosine Similarity' if dist_type=='cos' else 'Euclidean Distance', fontsize=FONTSIZE);
    ax.legend(ncol=4,bbox_to_anchor=(.5, .29), loc='upper center', shadow=True, prop=legend_settings).texts[0].set_text('Aug %');
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig('Figures/Paper/approximate_ws_{data}_{aug_type}.{save_type}'.format(
            data='cifar' if cifar else 'mnist',
            aug_type=aug_type,
            save_type=SAVE_TYPE
        ))
    else: plt.show()


def translation_paper_fig(fcn, cnn, cifar=False, save=True):
    data = 'cifar' if cifar else 'mnist'

    plt.clf(); fig=plt.figure(figsize=(20, FIG_HEIGHT*3))
    plt.subplot(3,2,1); aug_eval(fcn, y=[0,1.05], eval_type='acc'); plt.title('(a) ' + 'Translation Augmented ' + titles['acc'], fontsize=FONTSIZE)
    plt.subplot(3,2,2); aug_eval(cnn, y=[0,1.05], eval_type='acc'); plt.title('(b) ' + 'Translation Augmented ' + titles['acc'], fontsize=FONTSIZE)
    plt.subplot(3,2,3); aug_eval(fcn, y=[0.2,0.8] if cifar else [0.9,1], eval_type='val_acc'); plt.title('(c) ' + titles['val_acc'], fontsize=FONTSIZE)
    plt.subplot(3,2,4); aug_eval(cnn, y=[0.2,0.8] if cifar else [0.9,1], eval_type='val_acc'); plt.title('(d) ' + titles['val_acc'], fontsize=FONTSIZE)
    plt.subplot(3,2,5); aug_eval(fcn, y=[0.2,0.8] if cifar else [0,1.05], eval_type='translation'); plt.title('(e) ' + titles['translation'], fontsize=FONTSIZE)
    plt.subplot(3,2,6); aug_eval(cnn, y=[0.2,0.8] if cifar else [0,1.05], eval_type='translation'); plt.title('(f) ' + titles['translation'], fontsize=FONTSIZE)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.suptitle('Trained On Translation Augmented {data}'.format(
        data=data.upper()
    ))

    if save:
        plt.savefig('Figures/Paper/{data}.{save_type}'.format(
            data=data,
            save_type = SAVE_TYPE
        ))
    else: plt.show()

sm_ylims = {
    'mnist':{
        'rotation': {
            'train': [0.9,1.],
            'val_acc': [0.9,1],
            'trans': [0., 0.45],
            'rotation':[0.9,1],
            'noise':[.7,1],
            'edge_noise':[0.9,1]
        },
        'noise': {
            'train': [0.9,1.],
            'val_acc': [0.9,1],
            'trans': [0., 0.45],
            'rotation':[.7,.9],
            'noise':[0.9,1],
            'edge_noise':[0.9,1]
        },
        'edge_noise': {
            'train': [0.9,1.],
            'val_acc': [0.9,1],
            'trans': [0., 0.45],
            'rotation':[.7,.9],
            'noise':[0.9,1],
            'edge_noise':[0.9,1]
        },
        'vcp': {
            'train': [0.,1.],
            'val_acc': [0.,1],
            'trans': [0., 1],
            'rotation':[0., 1],
            'noise':[0., 1],
            'edge_noise':[0., 1]
        }
    },
    'cifar':{
        'rotation': {
            'train': [0.2,1],
            'val_acc': [0.2,.8],
            'trans': [0.2, .7],
            'rotation':[0.2,.7],
            'noise':[0.2,.7],
            'edge_noise':[0.2,.7]
        },
        'noise': {
            'train': [0.2,1],
            'val_acc': [0.2,.8],
            'trans': [0.1, .7],
            'rotation':[0.1,.7],
            'noise':[0.1,.7],
            'edge_noise':[0.1,.7]
        },
        'edge_noise': {
            'train': [0.2,1],
            'val_acc': [0.2,.8],
            'trans': [0.2, .7],
            'rotation':[0.2,.7],
            'noise':[0.2,.7],
            'edge_noise':[0.2,.7]
        },
        'vcp': {
            'train': [0.2,1],
            'val_acc': [0.2,.8],
            'trans': [0., .8],
            'rotation':[0.,.8],
            'noise':[0.,.8],
            'edge_noise':[0.,.8]
        }
    }
}


def translation_sm_fig(df, aug_type, cifar=False, save=True):
    data = 'cifar' if cifar else 'mnist'
    model = df['model'].unique()[0]

    plt.clf(); fig=plt.figure(figsize=(20, FIG_HEIGHT*3))
    plt.subplot(3,2,1); aug_eval(df, aug_type=aug_type, eval_type='acc', y=sm_ylims[data][aug_type]['train']); plt.title('(a) ' + titles['acc'], fontsize=FONTSIZE)
    plt.subplot(3,2,2); aug_eval(df, aug_type=aug_type, eval_type='val_acc', y=sm_ylims[data][aug_type]['val_acc']); plt.title('(b) ' + titles['val_acc'], fontsize=FONTSIZE)
    plt.subplot(3,2,3); aug_eval(df, aug_type=aug_type, eval_type='translation', y=sm_ylims[data][aug_type]['trans']); plt.title('(c) ' + titles['translation'], fontsize=FONTSIZE)
    plt.subplot(3,2,4); aug_eval(df, aug_type=aug_type, eval_type='rotation', y=sm_ylims[data][aug_type]['rotation']); plt.title('(d) ' + titles['rotation'], fontsize=FONTSIZE)
    plt.subplot(3,2,5); aug_eval(df, aug_type=aug_type, eval_type='noise', y=sm_ylims[data][aug_type]['noise']); plt.title('(e) ' + titles['noise'], fontsize=FONTSIZE)
    plt.subplot(3,2,6); aug_eval(df, aug_type=aug_type, eval_type='edge_noise', y=sm_ylims[data][aug_type]['edge_noise']); plt.title('(f) ' + titles['edge_noise'], fontsize=FONTSIZE)
    #plt.subplot(4,2,7); aug_eval(df, aug_type=aug_type, eval_type='swap', y=[0.2,0.8] if cifar else [0,1.05])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if aug_type == 'vcp':
        suptitle = 'VCP FCN Trained On Translation Augmented ' + data.upper()
    else:
        suptitle = '{model} Trained On {aug_type} {data}'.format(
            model=model.upper(),
            aug_type=titles[aug_type].replace(' Validation Accuracy',''),
            data=data.upper()
        )

    fig.suptitle(suptitle)

    if save:
        plt.savefig('Figures/SM/{model}_{aug_type}_{data}.{save_type}'.format(
            model=model,
            aug_type=aug_type,
            data=data,
            save_type=SAVE_TYPE
        ))
    else: plt.show()


def rotate(x, aug):
    datagen = ImageDataGenerator()

    return datagen.apply_transform(x,{'theta':int(aug*360)})

def trans(x, aug):
    datagen = ImageDataGenerator()

    return datagen.apply_transform(x,{'ty':aug*28})

def swap(x):
    tmp_x = deepcopy(x)
    hw = x.shape[1] / 2

    tmp_x[:hw, -hw:] = x[-hw:, :hw]
    tmp_x[-hw:, :hw] = x[:hw, -hw:]
    return tmp_x

def noise(x,noise_level):
    noise = np.random.normal(scale=noise_level*255., size=x.shape)

    return x + noise

def edge_noise(x,noise_level):
    noise = np.random.normal(scale=noise_level*255., size=x.shape)
    noise[5:-5, 5:-5, :] = 0

    return x + noise

def show_aug(aug_fn):

    plt.clf(); fig = plt.figure(figsize=(20,FIG_HEIGHT))
    if aug_fn.__name__ == 'swap':
        plt.subplot(1,2,1); plt.imshow(x); plt.axis('off'); plt.title('(a) Un-augmented')
        plt.subplot(1,2,2); plt.imshow(aug_fn(x)); plt.axis('off'); plt.title('(b) Quadrant Swap')
        plt.subplots_adjust(wspace=-.5, hspace=0)
        plt.tight_layout()
    else:
        gs = GridSpec(2, 7, figure=fig)
        for i, aug in enumerate([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, .99]):
            if 'noise' in aug_fn.__name__:
                title = r'(%s) $\mathcal{N}(0, %s)$' % (chr(97+i), str(aug))
            else:
                title = '({}) {} \%'.format(chr(97+i), 100*aug)

            x_trans = aug_fn(x[:,:,np.newaxis], aug).squeeze()

            if i == 0:
                ax = fig.add_subplot(gs[:, 0:2])
                ax.imshow(x); ax.axis('off');
            else:
                if i < 6: ax = fig.add_subplot(gs[0, i+1])
                else: ax = fig.add_subplot(gs[1, i-5+1])
                ax.imshow(x_trans); ax.axis('off');
            ax.set_title(title)
    plt.savefig('Figures/Augmentation/{name}.{save_type}'.format(
            name=aug_fn.__name__,
            save_type=SAVE_TYPE
        ))

    # cmap='Greys'


def plot_swap():
    df = pd.read_csv('mnist_1.csv')
    cnn = df[df['model'] == 'cnn']
    fcn = df[df['model'] == 'fcn']

    plt.clf(); _=plt.figure(figsize=(10, FIG_HEIGHT));

    p = sns.color_palette('coolwarm', 11)
    sns.lineplot(x='epoch', y='swap', data=cnn, label='CNN',color=p[0]); sns.lineplot(x='epoch', y='swap', data=fcn, label='FCN',color=p[-1])
    plt.title('Quadrant Swap MNIST'); plt.xlabel('Epochs', fontsize=FONTSIZE); plt.ylabel('Accuracy', fontsize=FONTSIZE); plt.legend(shadow=True, title='Model', prop=legend_settings, fontsize=FONTSIZE);

    plt.tight_layout(); plt.savefig('Figures/Paper/swap.{save_type}'.format(
        save_type=SAVE_TYPE
    ))
