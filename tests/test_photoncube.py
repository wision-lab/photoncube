from natsort import natsorted
import pytest
import numpy as np
import imageio.v3 as iio


@pytest.fixture
def grayscale_cube(tmp_path):
    path = tmp_path / "cube.npy"
    cube = np.random.randint(low=0, high=255, size=(256, 1024, 1024 // 8), dtype=np.uint8)
    np.save(path, cube)
    return path, cube


@pytest.fixture
def color_cube(tmp_path):
    path = tmp_path / "cube.npy"
    cube = np.random.randint(low=0, high=255, size=(256, 1024, 1024  // 8, 3), dtype=np.uint8)
    np.save(path, cube)
    return path, cube


def test_import():
    import photoncube
    from photoncube import PhotonCube, Transform


@pytest.mark.parametrize("cube_fixture", ("grayscale_cube", "color_cube"))
def test_open(cube_fixture, request):
    from photoncube import PhotonCube

    path, cube = request.getfixturevalue(cube_fixture)
    pc = PhotonCube.open(path)
    assert len(pc) == len(cube)
    assert pc.shape == cube.shape


@pytest.mark.parametrize("cube_fixture", ("grayscale_cube", "color_cube"))
def test_save_images(cube_fixture, tmp_path, request):
    from photoncube import PhotonCube

    path, cube = request.getfixturevalue(cube_fixture)
    pc = PhotonCube.open(path)
    pc.set_range(200, 205, 1)
    pc.save_images(tmp_path)

    for i, path in zip(range(200, 205), natsorted(tmp_path.glob("*.png"))):
        arr = iio.imread(path)
        packed = np.packbits(arr, axis=1)

        if cube_fixture == "grayscale_cube":
            assert np.allclose(packed.mean(axis=2), cube[i])
        else:
            assert np.allclose(packed, cube[i])


@pytest.mark.parametrize("cube_fixture", ("grayscale_cube", "color_cube"))
def test_slice(cube_fixture, request):
    from photoncube import PhotonCube

    path, cube = request.getfixturevalue(cube_fixture)
    pc = PhotonCube.open(path)

    assert np.allclose(pc[0], np.unpackbits(cube[0], axis=1))
    assert np.allclose(pc[-1], np.unpackbits(cube[-1], axis=1))


@pytest.mark.parametrize("cube_fixture", ("grayscale_cube", "color_cube"))
def test_getitem_python(cube_fixture, request, benchmark):
    path, _ = request.getfixturevalue(cube_fixture)
    cube = np.load(path, mmap_mode="r")
    benchmark(lambda: np.unpackbits(cube[0], axis=1))


@pytest.mark.parametrize("cube_fixture", ("grayscale_cube", "color_cube"))
def test_getitem_rust(cube_fixture, request, benchmark):
    from photoncube import PhotonCube

    path, _ = request.getfixturevalue(cube_fixture)
    pc = PhotonCube.open(path)
    benchmark(lambda: pc[0])


@pytest.mark.parametrize("cube_fixture", ("grayscale_cube", "color_cube"))
def test_masks_readonly(cube_fixture, request):
    from photoncube import PhotonCube

    path, _ = request.getfixturevalue(cube_fixture)
    pc = PhotonCube.open(path)

    assert pc.inpaint_mask == None
    assert pc.cfa_mask == None

    pc.load_mask("rgbw_oh_bn_color_ss2_corrected.png")

    assert not pc.inpaint_mask.flags["WRITEABLE"]

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        pc.inpaint_mask[0, 0] = False


def test_validate_transforms():
    from photoncube import Transform

    with pytest.raises(RuntimeError, match="invalid variant"):
        Transform.from_str("abc")
