import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import contextlib
import io

from PIL import Image

SRC=Path(__file__).resolve().parents[1]/"src"
sys.path.insert(0,str(SRC))
from fashioniq_noise import (audit,compile_policy,dataset_fingerprint,image_path,
    load_noise_policy,object_sha,review_template,text_flags,validate_policy)
from review_fashioniq_triplets import validate_judgment


class NoiseTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="fashioncir-noise-test-")
        self.root=Path(self.temp.name).resolve()
        if self.root.parent != Path(tempfile.gettempdir()).resolve() or not self.root.name.startswith("fashioncir-noise-test-"):
            raise ValueError("Unexpected test cleanup target")
        self.addCleanup(self.temp.cleanup)
        for name in ("captions","image_splits","resized_image"):
            (self.root/name).mkdir()
        self.rows=[
            dict(candidate="ref",target="a",captions=["a shirt with a logo","has printed text"]),
            dict(candidate="ref",target="logo",captions=["is a floral dress","is a company name and not a dress"]),
            dict(candidate="logo",target="b",captions=["is blue","has sleeves"]),
            dict(candidate="ref",target="missing",captions=["is blue","is longer"]),
            dict(candidate="corrupt",target="b",captions=["is blue","is longer"]),
            dict(candidate="ref",target="b",captions=["","is longer"]),
            dict(candidate="ref",target="ref",captions=["is blue","is longer"]),
            dict(candidate="ref",target="a",captions=["a shirt with a logo","has printed text"]),
            dict(candidate="ref",target="b",captions=["is solid blue","is solid red"]),
            dict(candidate="ref",target="b",captions=["is blue","has a collar"]),
        ]
        self.caption_path=self.root/"captions/cap.dress.train.json"
        self.write(self.caption_path,self.rows)
        self.write(self.root/"captions/cap.dress.val.json",[self.rows[0]])
        self.write(self.root/"image_splits/split.dress.train.json",["ref","a","b","logo","missing","corrupt"])
        self.write(self.root/"image_splits/split.dress.val.json",["ref","a"])
        self.write(self.root/"captions/image_captions_dress_train.json",{"logo":"a logo on a white background","a":"a shirt with a logo"})
        for name in ("ref","a","b","logo"):
            Image.new("RGB",(40,40),"blue").save(self.root/f"resized_image/{name}.jpg")
        (self.root/"resized_image/corrupt.jpg").write_bytes(b"not an image")
        self.report=audit(self.root,"dress")

    @staticmethod
    def write(path,value):
        Path(path).write_text(json.dumps(value),encoding="utf-8")

    def test_hard_file_failures_are_detected_and_pending_semantics_kept(self):
        self.assertEqual(self.report["images"]["missing"]["hard_errors"],["missing_image"])
        self.assertEqual(self.report["images"]["corrupt"]["hard_errors"],["unreadable_image"])
        # Inject a model screening hint. It must not become an automatic exclusion.
        self.report["images"]["logo"]["flags"].append("clip_non_garment_candidate")
        policy=compile_policy(self.root,self.report,review_template(self.report))
        excluded,blocked=validate_policy(self.root,"dress",policy)
        self.assertEqual(excluded,{3,4})
        self.assertEqual(blocked,{"missing","corrupt"})

    def test_printed_logo_and_ambiguous_captions_are_not_rejected(self):
        self.assertNotIn("caption_non_garment_claim",text_flags(self.rows[0]["captions"]))
        self.assertIn("caption_domain_disagreement",text_flags(self.rows[1]["captions"]))
        self.assertIn("caption_color_disagreement",text_flags(self.rows[8]["captions"]))
        self.assertTrue(self.report["exact_duplicate_groups"])
        self.assertEqual(len(compile_policy(self.root,self.report)["excluded_queries"]),2)

    def test_reviewed_image_excludes_every_incident_query(self):
        review=review_template(self.report)
        row=next(r for r in review["images"] if r["image_id"]=="logo")
        row.update(decision="exclude",reviewer="test reviewer",reason="standalone wordmark, no garment")
        policy=compile_policy(self.root,self.report,review,.5)
        excluded,_=validate_policy(self.root,"dress",policy)
        self.assertEqual(excluded,{1,2,3,4})
        tampered=copy.deepcopy(policy)
        tampered["excluded_queries"]=[r for r in tampered["excluded_queries"] if r["source_index"]!=2]
        with self.assertRaises(ValueError): validate_policy(self.root,"dress",tampered)

    def test_query_only_review_preserves_image_in_other_queries(self):
        review=review_template(self.report)
        row=next(r for r in review["queries"] if r["source_index"]==1)
        row.update(decision="exclude",reviewer="test reviewer",reason="captions disagree")
        policy=compile_policy(self.root,self.report,review,.5)
        excluded,blocked=validate_policy(self.root,"dress",policy)
        self.assertIn(1,excluded)
        self.assertNotIn(2,excluded)
        self.assertNotIn("logo",blocked)

    def test_review_requires_identity_evidence_and_reviewer(self):
        for change in [dict(decision="exclude"),dict(sha256="stale"),dict(decision="bogus")]:
            review=review_template(self.report)
            review["images"][0].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                compile_policy(self.root,self.report,review,.9)
        review=review_template(self.report)
        review["queries"].append(review["queries"][0])
        with self.assertRaises(ValueError): compile_policy(self.root,self.report,review,.9)

    def test_stale_annotations_and_image_repairs_require_reaudit(self):
        policy=compile_policy(self.root,self.report)
        Image.new("RGB",(40,40)).save(self.root/"resized_image/missing.jpg")
        with self.assertRaises(ValueError): validate_policy(self.root,"dress",policy)
        self.rows.reverse()
        self.write(self.caption_path,self.rows)
        with self.assertRaises(ValueError): compile_policy(self.root,self.report)
        with self.assertRaises(ValueError): validate_policy(self.root,"dress",policy)

    def test_partial_and_validation_reports_cannot_filter_train(self):
        for report in [dict(self.report,partial=True),dict(self.report,split="val")]:
            with self.assertRaises(ValueError): compile_policy(self.root,report)
        policy=compile_policy(self.root,self.report)
        policy["split"]="val"
        with self.assertRaises(ValueError): validate_policy(self.root,"dress",policy)
        with self.assertRaises(ValueError): compile_policy(self.root,self.report,max_exclusion_fraction=.1)

    def test_default_off_and_nested_image_paths(self):
        self.assertEqual(load_noise_policy("/does/not/exist","dress"),(set(),set()))
        self.assertIsNone(image_path(self.root,"dress","../outside"))
        nested=self.root/"resized_image/dress"
        nested.mkdir()
        (self.root/"resized_image/a.jpg").rename(nested/"a.jpg")
        self.assertEqual(image_path(self.root,"dress","a"),nested/"a.jpg")
        with self.assertRaises(ValueError): validate_policy(self.root,"dress",self.report)

    def test_semantic_dry_run_and_unavailable_model_produce_no_labels(self):
        import review_fashioniq_triplets as runner
        requests=self.root/"requests.jsonl"
        output=self.root/"proposals.json"
        requests.write_text(json.dumps(dict(source_index=0,row_sha256=object_sha(self.rows[0]),
            category="dress",split="train",dataset_fingerprint=dataset_fingerprint(self.root,"dress"),
            reference_path=str(self.root/"resized_image/ref.jpg"),target_path=str(self.root/"resized_image/a.jpg"),
            prompt="Judge both images"))+"\n",encoding="utf-8")
        argv=["review_fashioniq_triplets.py","--requests",str(requests),"--output",str(output)]
        with patch.object(sys,"argv",argv+["--dry-run"]),contextlib.redirect_stdout(io.StringIO()) as stream:
            runner.main()
        self.assertEqual(len(json.loads(stream.getvalue())["previews"][0]["image_order"]),2)
        self.assertFalse(output.exists())
        with patch.object(sys,"argv",argv),patch("generate_structured_edits.load_qwen",side_effect=OSError("missing offline weights")):
            with self.assertRaises(SystemExit): runner.main()
        payload=json.loads(output.read_text())
        self.assertEqual(payload["state"],"model_unavailable")
        self.assertEqual(payload["samples"],[])

    def test_malformed_annotations_excluded_before_loader_access(self):
        self.rows[5]={"broken":True}
        self.write(self.caption_path,self.rows)
        report=audit(self.root,"dress")
        policy=compile_policy(self.root,report,max_exclusion_fraction=.5)
        excluded,_=validate_policy(self.root,"dress",policy)
        self.assertIn(5,excluded)
        def method(file,name,clsname):
            tree=ast.parse((SRC/file).read_text(encoding="utf-8"))
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==clsname)
            node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
            namespace={"os":__import__("os")}
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),"exec"),namespace)
            return namespace[name]
        fake=types.SimpleNamespace(caption_dir=str(self.root/"captions"),category="dress",
            _load_json=lambda path,**kw:json.loads(Path(path).read_text()),
            concat_text=lambda caps:" and ".join(caps),noise_excluded_indices=excluded)
        load=method("newDataset.py","_load_queries","FashionIQ")
        rows=load(fake,"train")
        self.assertNotIn(5,{r["source_index"] for r in rows})
        self.assertEqual(len(load(fake,"val")),1)  # validation is not filtered
        fake.caption_dir=Path(fake.caption_dir)
        fake._concat_text=fake.concat_text
        baseline=method("dquDataset.py","_build_train_data","FashionIQ")
        self.assertEqual(len(baseline(fake)),len(rows))
        # Sparse source indices survive; filtering must not renumber reviewed labels.
        self.assertEqual([r['source_index'] for r in rows],[i for i in range(10) if i not in excluded])


class SemanticSchemaTest(unittest.TestCase):
    def test_uncertainty_is_valid_and_model_output_is_not_a_filter(self):
        value=dict(reference_kind="garment",target_kind="unknown",recommendation="uncertain",captions_conflict=None,
            evidence="target attribute not visible",caption_judgments=[dict(index=i,status="unknown",evidence="not visible") for i in range(2)])
        self.assertEqual(validate_judgment(value),value)
        for change in [dict(reference_kind=[]),dict(recommendation=[]),dict(captions_conflict="false"),
                       dict(caption_judgments=[value['caption_judgments'][0]]),dict(evidence="")]:
            with self.subTest(change=change),self.assertRaises(ValueError): validate_judgment(dict(value,**change))


if __name__=="__main__": unittest.main()
