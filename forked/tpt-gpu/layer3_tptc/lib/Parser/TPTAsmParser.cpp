#include "../../include/tptir/Parser/TPTAsmParser.h"
#include <cctype>
namespace tptir {
Lexer::Lexer(const std::string& i) : input_(i), pos_(0), line_(1), column_(1), tokenPos_(0) {}
bool Lexer::isAtEnd() const { return pos_ >= input_.size(); }
char Lexer::peek() const { return isAtEnd() ? '\0' : input_[pos_]; }
char Lexer::advance() { char c = input_[pos_++]; column_++; return c; }
void Lexer::skipWhitespace() { while (!isAtEnd() && std::isspace(peek())) { if (peek() == '\n') { line_++; column_=1; } advance(); } }
void Lexer::skipComment() { while (!isAtEnd() && peek() != '\n') advance(); }
Token Lexer::scanIdentifier() {
  size_t start=pos_,col=column_;
  while(!isAtEnd()&&(std::isalnum(peek())||peek()=='_'||peek()=='.'))advance();
  std::string text=input_.substr(start,pos_-start);
  TokenKind kind=TokenKind::Identifier;
  if(text=="module")kind=TokenKind::KwModule;
  else if(text=="func")kind=TokenKind::KwFunc;
  else if(text=="memref")kind=TokenKind::KwMemref;
  else if(text=="tensor")kind=TokenKind::KwTensor;
  return{kind,text,line_,col};
}
Token Lexer::scanNumber() {
  size_t start=pos_,col=column_;bool isFloat=false;
  while(!isAtEnd()&&(std::isdigit(peek())||peek()=='.')){if(peek()=='.')isFloat=true;advance();}
  return{isFloat?TokenKind::Float:TokenKind::Integer,input_.substr(start,pos_-start),line_,col};
}
void Lexer::tokenizeAll(){
  while(!isAtEnd()){
    skipWhitespace();if(isAtEnd())break;
    char c=peek();
    if(c==';'||c=='#'){skipComment();continue;}
    if(std::isalpha(c)||c=='_'){tokens_.push_back(scanIdentifier());continue;}
    if(std::isdigit(c)){tokens_.push_back(scanNumber());continue;}
    advance();
    switch(c){
      case'(':tokens_.push_back({TokenKind::LParen,"(",line_,column_-1});break;
      case')':tokens_.push_back({TokenKind::RParen,")",line_,column_-1});break;
      case'{':tokens_.push_back({TokenKind::LBrace,"{",line_,column_-1});break;
      case'}':tokens_.push_back({TokenKind::RBrace,"}",line_,column_-1});break;
      case',':tokens_.push_back({TokenKind::Comma,",",line_,column_-1});break;
      case':':tokens_.push_back({TokenKind::Colon,":",line_,column_-1});break;
      case'<':tokens_.push_back({TokenKind::Less,"<",line_,column_-1});break;
      case'>':tokens_.push_back({TokenKind::Greater,">",line_,column_-1});break;
      case'=':tokens_.push_back({TokenKind::Equals,"=",line_,column_-1});break;
      case'@':tokens_.push_back({TokenKind::At,"@",line_,column_-1});break;
      case'+':tokens_.push_back({TokenKind::Plus,"+",line_,column_-1});break;
      case'-':if(peek()=='>'){advance();tokens_.push_back({TokenKind::Arrow,"->",line_,column_-2});}else tokens_.push_back({TokenKind::Minus,"-",line_,column_-1});break;
      case'*':tokens_.push_back({TokenKind::Star,"*",line_,column_-1});break;
      default:tokens_.push_back({TokenKind::Unknown,std::string(1,c),line_,column_-1});break;
    }
  }
  tokens_.push_back({TokenKind::Eof,"",line_,column_});
}
Token Lexer::nextToken(){if(tokenPos_>=tokens_.size())tokenizeAll();return tokenPos_<tokens_.size()?tokens_[tokenPos_++]:Token{TokenKind::Eof,"",line_,column_};}
Token Lexer::peekToken(){if(tokenPos_>=tokens_.size())tokenizeAll();return tokenPos_<tokens_.size()?tokens_[tokenPos_]:Token{TokenKind::Eof,"",line_,column_};}
TPTAsmParser::TPTAsmParser(const std::string& i):lexer_(i),pos_(0),nextValueId_(0){lexer_.tokenizeAll();tokens_=lexer_.tokens();}
Token TPTAsmParser::peek() const{return pos_<tokens_.size()?tokens_[pos_]:Token{TokenKind::Eof,"",0,0};}
Token TPTAsmParser::advance(){return pos_<tokens_.size()?tokens_[pos_++]:Token{TokenKind::Eof,"",0,0};}
bool TPTAsmParser::match(TokenKind k){if(check(k)){advance();return true;}return false;}
bool TPTAsmParser::check(TokenKind k) const{return peek().kind==k;}
Type* TPTAsmParser::parseType(){auto tok=advance();return parseType(tok.text);}
Region* TPTAsmParser::parseFunction(){auto*r=new Region();r->addBlock(new Block("entry"));return r;}
std::vector<Block*> TPTAsmParser::parseModule(){return{new Block("entry")};}
}
