import pytest
import graphics.bone_palette_fence as mod

CASES = [
    ("capacity", (1,)),
    ("chunk_split", (1,)),
    ("upload_fence", (1,)),
    ("generation", (1,)),
    ("delayed_release", (1,)),
    ("index_align", (1,)),
    ("weight_renorm", (1,)),
    ("matrix_reduce", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("metrics", (1,)),
    ("determinism", (1,)),
    ("reload", (1,)),
    ("profile", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
