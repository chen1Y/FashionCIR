"""New021 generation version; retain020 stop; unchanged sealed020 training design."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_expanded_attr_020 as core
from expanded_attr_provider_021 import generate
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

core.generate=generate
core.FILES=core.FILES+('src/expanded_attr_schema_021.py','src/expanded_attr_provider_021.py','scripts/run_expanded_attr_021.py','tests/test_expanded_attr_021.py')

if __name__=='__main__':
    core.main()
    if '--prepare' in sys.argv:
        root=Path(sys.argv[sys.argv.index('--root')+1]);old=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-020')
        check=json.loads((old/'state.json').read_text())
        if check['status']!='failed' or check['error']!='020 batch stopped; preserve known/unknown and do not retry':raise ValueError('Unexpected020 failure; inspect before021')
        plan=json.loads((root/'plan.json').read_text());plan.update(id='expanded_attr021',generation_version='021',first_paid_batch_queries=8,parent020_failed=True,parent020_state_sha256=file_sha(old/'state.json'),unchanged020_training_protocol=True)
        for name in ('state.json','generation/train_000/responses.json','generation/train_000/protocol.json','generation/train_000/completion.json'):
            p=old/name;st=p.stat();plan['protected_texts'][str(p)]=dict(sha256=file_sha(p),size=st.st_size,mtime_ns=st.st_mtime_ns)
        atomic_write_json(root/'plan.json',plan)
        native=json.loads((old/'native_preflight.json').read_text());native['scope']+=' Carried architecture check; each021 arm must also pass new-data native smoke before fitting.'
        atomic_write_json(root/'native_preflight.json',native)
