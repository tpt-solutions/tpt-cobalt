#ifndef TPTIR_PARSER_TPTASMPARSER_H
#define TPTIR_PARSER_TPTASMPARSER_H
#include "../Dialect/TPTIRDialect.h"
#include "../Dialect/TPTIROps.h"
namespace tptir {
enum class TokenKind : uint8_t {
  Identifier, Integer, Float, String,
  LParen, RParen, LBrace, RBrace, LBracket, RBracket,
  Less, Greater, Comma, Colon, Arrow, Equals, At,
  Plus, Minus, Star, Slash, Percent,
  KwModule, KwFunc, KwMemref, KwTensor, KwVector,
  KwGlobal, KwShared, KwLocal, KwConstant, Eof, Unknown
};
struct Token { TokenKind kind; std::string text; size_t line; size_t column; };
class Lexer {
public:
  explicit Lexer(const std::string& input);
  Token nextToken();
  Token peekToken();
  const std::vector<Token>& tokens() const { return tokens_; }
  void tokenizeAll();
private:
  void skipWhitespace(); void skipComment();
  Token scanIdentifier(); Token scanNumber();
  char peek() const; char advance(); bool isAtEnd() const;
  std::string input_;
  size_t pos_{0}, line_{1}, column_{1}, tokenPos_{0};
  std::vector<Token> tokens_;
};
class TPTAsmParser {
public:
  explicit TPTAsmParser(const std::string& input);
  Region* parseFunction();
  std::vector<Block*> parseModule();
  const std::string& lastError() const { return lastError_; }
  const std::string& functionName() const { return funcName_; }
private:
  Token consume(TokenKind expected, const std::string& errMsg);
  Token peek() const; Token advance(); bool match(TokenKind kind);
  Type* parseType(); std::vector<int64_t> parseShape();
  AddressSpace parseAddressSpace();
  Block* parseBlock(); Operation* parseOperation();
  Value* parseValue(); std::string parseIdentifier();
  Lexer lexer_; std::vector<Token> tokens_; size_t pos_{0};
  std::string lastError_, funcName_;
  std::vector<std::pair<std::string, Type*>> funcArgs_;
  uint64_t nextValueId_{0};
  std::unordered_map<std::string, Value*> valueMap_;
  std::unordered_map<std::string, Block*> blockMap_;
};
}
#endif
