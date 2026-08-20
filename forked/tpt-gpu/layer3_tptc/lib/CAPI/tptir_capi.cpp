#include "../../include/tptir/CAPI/tptir_capi.h"
#include "../../include/tptir/Dialect/TPTIRDialect.h"
#include "../../include/tptir/Parser/TPTAsmParser.h"
#include "../../include/tptir/Pass/TPTIRPasses.h"
#include "../../include/tptir/CodeGen/TPTCodeGen.h"
#include <cstring>
#include <cstdlib>
struct Ctx{bool ok;Ctx():ok(true){tptir::registerTPTIRDialect();}};
extern "C" {
tptir_status_t tptir_init(tptir_context_t* c){if(!c)return TPTIR_ERROR_NULL_POINTER;*c=new Ctx();return TPTIR_OK;}
tptir_status_t tptir_shutdown(tptir_context_t c){if(!c)return TPTIR_ERROR_NULL_POINTER;delete static_cast<Ctx*>(c);return TPTIR_OK;}
tptir_version_t tptir_get_version(){return{0,1,0};}
const char* tptir_status_string(tptir_status_t s){switch(s){case TPTIR_OK:return"OK";case TPTIR_ERROR_PARSE:return"Parse error";case TPTIR_ERROR_NULL_POINTER:return"Null pointer";default:return"Error";}}
tptir_status_t tptir_module_create(tptir_context_t c,tptir_module_t* m){if(!c||!m)return TPTIR_ERROR_NULL_POINTER;*m=new tptir::Region();return TPTIR_OK;}
tptir_status_t tptir_module_destroy(tptir_module_t m){if(!m)return TPTIR_ERROR_NULL_POINTER;delete static_cast<tptir::Region*>(m);return TPTIR_OK;}
tptir_status_t tptir_parser_create(tptir_context_t c,const char* s,size_t l,tptir_parser_t* p){if(!c||!s||!p)return TPTIR_ERROR_NULL_POINTER;*p=new tptir::TPTAsmParser(std::string(s,l));return TPTIR_OK;}
tptir_status_t tptir_parser_destroy(tptir_parser_t p){if(!p)return TPTIR_ERROR_NULL_POINTER;delete static_cast<tptir::TPTAsmParser*>(p);return TPTIR_OK;}
tptir_status_t tptir_parser_parse_function(tptir_parser_t p,tptir_region_t* r){if(!p||!r)return TPTIR_ERROR_NULL_POINTER;*r=static_cast<tptir::TPTAsmParser*>(p)->parseFunction();return*r?TPTIR_OK:TPTIR_ERROR_PARSE;}
tptir_status_t tptir_pass_pipeline_create(tptir_context_t c,tptir_pass_pipeline_t* p){if(!c||!p)return TPTIR_ERROR_NULL_POINTER;*p=tptir::createDefaultPassPipeline();return TPTIR_OK;}
tptir_status_t tptir_pass_pipeline_destroy(tptir_pass_pipeline_t p){if(!p)return TPTIR_ERROR_NULL_POINTER;delete static_cast<tptir::PassPipeline*>(p);return TPTIR_OK;}
tptir_status_t tptir_pass_pipeline_run(tptir_pass_pipeline_t p,tptir_region_t r,uint64_t* n){if(!p||!r)return TPTIR_ERROR_NULL_POINTER;if(n)*n=static_cast<tptir::PassPipeline*>(p)->run(static_cast<tptir::Region*>(r));return TPTIR_OK;}
tptir_status_t tptir_codegen_create(tptir_context_t c,int32_t t,tptir_codegen_t* g){if(!c||!g)return TPTIR_ERROR_NULL_POINTER;tptir::CodeGenOptions o;o.target=static_cast<tptir::CodeGenTarget>(t);*g=new tptir::TPTCodeGen(o);return TPTIR_OK;}
tptir_status_t tptir_codegen_destroy(tptir_codegen_t g){if(!g)return TPTIR_ERROR_NULL_POINTER;delete static_cast<tptir::TPTCodeGen*>(g);return TPTIR_OK;}
tptir_status_t tptir_codegen_generate(tptir_codegen_t g,tptir_region_t r,tptir_string_t* o){if(!g||!r||!o)return TPTIR_ERROR_NULL_POINTER;auto s=static_cast<tptir::TPTCodeGen*>(g)->generate(static_cast<tptir::Region*>(r));o->size=s.size();o->data=(char*)std::malloc(s.size()+1);if(o->data)std::memcpy(o->data,s.c_str(),s.size()+1);return TPTIR_OK;}
void tptir_string_free(tptir_string_t* s){if(s&&s->data){std::free(s->data);s->data=nullptr;s->size=0;}}
tptir_status_t tptir_compile(const char* s,size_t l,int32_t t,tptir_string_t* o,tptir_string_t* e){try{auto p=new tptir::TPTAsmParser(std::string(s,l));auto r=p->parseFunction();tptir::CodeGenOptions opts;opts.target=static_cast<tptir::CodeGenTarget>(t);auto cg=new tptir::TPTCodeGen(opts);auto result=cg->generate(r);o->size=result.size();o->data=(char*)std::malloc(result.size()+1);if(o->data)std::memcpy(o->data,result.c_str(),result.size()+1);delete cg;delete p;delete r;return TPTIR_OK;}catch(...){return TPTIR_ERROR_GENERIC;}}
}
