import tara
from glob import glob
import os

data_dir = tara.data_dir

exps = glob(f'{data_dir}/pista_sim/*.fits')

def test_init():

    tar_obj = tara.tara(exps)

    assert hasattr(tar_obj, 'exps')
    assert hasattr(tar_obj, 'shape')
    assert tar_obj.shape[0]==300

def test_show_image():
    tar_obj = tara.tara(exps)

    _,_, phot_table = tar_obj.show_image()

    assert 'flux' in phot_table.keys()
    assert 'SNR' in phot_table.keys()

def test_call():
    if not os.path.exists('test_out'):
        os.mkdir('test_out')
    tar_obj = tara.tara(exps, out_dir='test_out')
    ref_pos, ref_img = tar_obj(rnge =[0,3],step=3)
    os.remove('test_out')