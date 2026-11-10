import unittest,tempfile,json
from pathlib import Path
from expanded_attr_resume_023 import provider_code,publish_ready

class SafeProviderDiagnostics(unittest.TestCase):
    def test_nested_code(self):self.assertEqual(provider_code({'error':{'code':'InvalidParameter','message':'private'}}),'InvalidParameter')
    def test_top_level_code(self):self.assertEqual(provider_code({'code':'DataInspectionFailed','message':'private'}),'DataInspectionFailed')
    def test_secret_or_free_text_refused(self):
        for x in ('sk-'+'x'*32,'Prefix:sk-'+'x'*32,'private user message','x'*81,'<image>'):
            self.assertEqual(provider_code({'code':x}),'unavailable')
    def test_unexpected_body(self):
        for x in (None,[],{},dict(error='private'),dict(code=123)):
            self.assertEqual(provider_code(x),'unavailable')
    def test_ready_category_does_not_create_empty_other_category(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            root=Path(t);rows=[dict(category='shirt',request_key='a',source_index=1,reference='ref',instruction='longer'),dict(category='toptee',request_key='b',source_index=2,reference='ref2',instruction='shorter')]
            answers={'a':dict(status='ok',target_attrs=dict(color='white'),errors=[])}
            publish_ready(root,'train',rows,answers);p=root/'texts/shirt_train.json';before=p.read_bytes()
            answers['b']=dict(status='fallback',target_attrs=dict(color=''),errors=['unknown_no_resend'])
            publish_ready(root,'train',rows,answers);self.assertEqual(p.read_bytes(),before);self.assertEqual(len(json.loads((root/'texts/toptee_train.json').read_text())['samples']),1)

if __name__=='__main__':unittest.main()
