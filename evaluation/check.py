"""Check file layout, imports, checkpoints and stored result hashes."""
import argparse,ast,hashlib,io,json,pickletools,tokenize,zipfile
from pathlib import Path
from model.catalog import ROOT,CATALOG
from model.checkpoint_pickle import MODULE_MAP
from other.integrity import verify_runtime_sources
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args(argv)
    verify_runtime_sources()
    spec=json.loads((ROOT/'other/provenance/source_manifest.json').read_text(encoding='utf8'))
    for rel,value in spec['data_sha256'].items():
        if sha(ROOT/rel)!=value:raise ValueError('Data changed: '+rel)
    pyfiles=[p for d in ('model','evaluation','analysis','other','data') for p in (ROOT/d).rglob('*.py')]
    definitions={}
    modules=set()
    for file in pyfiles:
        source=file.read_text(encoding='utf-8-sig')
        tree=ast.parse(source,filename=str(file));compile(tree,str(file),'exec')
        module='.'.join(file.relative_to(ROOT).with_suffix('').parts).removesuffix('.__init__')
        modules.add(module)
        definitions[module]={n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        if 'data' not in file.relative_to(ROOT).parts[:1]:
            comments=[t.string for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type==tokenize.COMMENT]
            if any(not c.isascii() for c in comments):raise ValueError('Non-English comment: '+str(file))
            for n in ast.walk(tree):
                if isinstance(n,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                    doc=ast.get_docstring(n)
                    if doc and not doc.isascii():raise ValueError('Non-English docstring: '+str(file))
    for file in pyfiles:
        tree=ast.parse(file.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom):
                parts=list(file.relative_to(ROOT).parent.parts)
                prefix=parts[:len(parts)-n.level+1] if n.level else []
                names=['.'.join(prefix+((n.module or '').split('.') if n.module else []))]
            else:continue
            for name in names:
                if name and name.split('.')[0] in ('model','evaluation','analysis','other','ref','thermal_revision') and name not in modules:
                    raise ValueError('Missing local module '+name+' in '+str(file))
    checkpoints=0
    for file in (ROOT/'model/checkpoints').rglob('*.pth'):
        with zipfile.ZipFile(file) as archive:
            member=next(n for n in archive.namelist() if n.endswith('/data.pkl'))
            data=archive.read(member)
        for op,arg,pos in pickletools.genops(data):
            if op.name!='GLOBAL':continue
            module,name=arg.split(' ',1)
            if not module.startswith(('model.','ref.')):continue
            if module=='model.household.legacy' and name=='Legacy':name='ContinuousDevicePolicy'
            module=MODULE_MAP.get(module,module)
            if name not in definitions.get(module,set()):raise ValueError('Unresolved checkpoint class '+module+'.'+name)
        checkpoints+=1
    if not args.quick:
        for rel,value in spec['artifact_sha256'].items():
            if sha(ROOT/rel)!=value:raise ValueError('Result or checkpoint changed: '+rel)
    for name,item in CATALOG.items():
        file=ROOT/Path(*item['module'].split('.')).with_suffix('.py')
        if not file.is_file() or not (ROOT/'evaluation'/(name+'.py')).is_file():raise ValueError('Missing algorithm entry '+name)
    print('[Passed]',len(pyfiles),'Python files,',len(CATALOG),'algorithms,',checkpoints,'checkpoints,',len(spec['data_sha256']),'data files')
if __name__=='__main__':main()
