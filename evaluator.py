import numpy as np
import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from utils.molecular_metrics import MolecularMetrics

def mols2grid_image(mols, molsPerRow, legends=None):
    mols = [e if e is not None else Chem.RWMol() for e in mols]
    if legends is not None:
        legends = [f'{le:.3f}' if e is not None else '-' for le, e in zip(legends, mols)]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150), legends=legends)


source = "results"

z_dim = 8

def main():
    # session
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    data = SparseMolecularDataset()
    data.load('data/gdb9_9nodes.sparsedataset')

    model = GraphGANModel(data.vertexes,
                      data.bond_num_types,
                      data.atom_num_types,
                      z_dim,
                      decoder_units=(128, 256, 512),
                      discriminator_units=((128, 64), 128, (128, 64)),
                      decoder=decoder_adj,
                      discriminator=encoder_rgcn,
                      soft_gumbel_softmax=False,
                      hard_gumbel_softmax=False,
                      batch_discriminator=False)

    saver = tf.train.Saver()

    print('load trained model')
    saver.restore(session, f'{source}/model.ckpt')

    # 乱数の生成
    embeddings = model.sample_z(10)

    # modelへ乱数を渡す
    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
            feed_dict={model.training: False, model.embeddings: embeddings})

    # argmaxxにより確率から値を生成
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    # ベクトルからグラフ
    mols = [data.matrices2mol(n_, e_, strict=True) for n_,e_ in zip(n, e)]

    qed = MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)

    n_col = 5
    img = mols2grid_image(mols, n_col, legends=qed)
    img.save("trained_model_mols.png")


if __name__ == "__main__":
    main()


