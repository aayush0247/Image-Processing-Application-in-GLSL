////////////////////////////////////////////////////////////////////////
// Image Processing Web App
var gl;
var canvas;

var backgroundImageInput;
var foregroundImageInput;
var matrixStack = [];

var aPositionLocation;
var aTexCoordLocation;
var uMMatrixLocation;
var uTextureLocation;
var uTexture2Location;
var uDiffuseTermLocation;
var uFlagLocation;
var uContrastLocation;
var uBrightLocation;

var sqVertices = [];
var sqIndices = [];
var sqNormals = [];
var sqTexCoords = [];
var sqVerPosBuf;
var sqVerIndexBuf;
var sqVerNormalBuf;
var sqVerTexCoordsBuf;

var flagtexture = 0;
var flag = [0.0];
var contrast = [0.0];
var brightness = [0.0];

var texture1, texture2;
var alphaTexture;
var alphaFBO;

var selectedImageType = "background";
var selectedEffectType = "sepia";

var mMatrix = mat4.create(); // model matrix

var eyePos = [0.0, 0.0, 8];

//////////////////////////////////////////////////////////////////////////
const vertexShaderCode = `#version 300 es
in vec3 aPosition;
in vec2 aTexCoords;

uniform mat4 uMMatrix;

out vec2 fragTexCoord;

void main() {
  // pass texture coordinate to frag shader
  fragTexCoord = aTexCoords;

  // calculate clip space position
  gl_Position =  uMMatrix * vec4(aPosition,1.0);
  gl_PointSize=3.0;
}`;

const fragShaderCode = `#version 300 es
precision mediump float;

in vec2 fragTexCoord;

uniform sampler2D imageTexture;
uniform sampler2D imageTexture2;

out vec4 fragColor;

void main() {
  fragColor = vec4(0,0,0,1);

  //look up texture color
  vec4 textureColor = texture(imageTexture, fragTexCoord);
  vec4 textureColor2 =  texture(imageTexture2, fragTexCoord);

  float alpha = textureColor.a;
  vec4 alphaBlend = alpha*textureColor + (1.0-alpha)*textureColor2;

  fragColor = alphaBlend;
}`;

const vertexShaderCode1 = `#version 300 es
in vec3 aPosition;
in vec2 aTexCoords;

uniform mat4 uMMatrix;

out vec2 fragTexCoord;

void main() {
  // pass texture coordinate to frag shader
  fragTexCoord = aTexCoords;

  // calculate clip space position
  gl_Position =  uMMatrix * vec4(aPosition,1.0);
  gl_PointSize=3.0;
}`;

const fragShaderCode1 = `#version 300 es
precision mediump float;

in vec2 fragTexCoord;

uniform sampler2D imageTexture;

out vec4 fragColor;

void main() {
  fragColor = vec4(0,0,0,1);

  // look up texture color
  vec4 textureColor =  texture(imageTexture, fragTexCoord);

  fragColor = textureColor;
}`;

const vertexShaderCode2 = `#version 300 es
in vec3 aPosition;
in vec2 aTexCoords;

uniform mat4 uMMatrix;

out vec2 fragTexCoord;

void main() {
  // pass texture coordinate to frag shader
  fragTexCoord = aTexCoords;

  // calculate clip space position
  gl_Position =  uMMatrix * vec4(aPosition,1.0);
  gl_PointSize=3.0;
}`;

const fragShaderCode2 = `#version 300 es
precision highp float;

in vec2 fragTexCoord;

uniform sampler2D imageTexture;

out vec4 fragColor;

void main() {
  fragColor = vec4(0,0,0,1);
}`;

const vertexShaderCode3 = `#version 300 es
in vec3 aPosition;
in vec2 aTexCoords;

uniform mat4 uMMatrix;

out vec2 fragTexCoord;

void main() {
  // pass texture coordinate to frag shader
  fragTexCoord = aTexCoords;

  // calculate clip space position
  gl_Position =  uMMatrix * vec4(aPosition,1.0);
  gl_PointSize=3.0;
}`;

const fragShaderCode3 = `#version 300 es
precision highp float;

in vec2 fragTexCoord;

uniform sampler2D imageTexture;
uniform float uFlag; 

out vec4 fragColor;

void main() {
  fragColor = vec4(0,0,0,1);
  
  // look up texture color
  vec4 textureColor =  texture(imageTexture, fragTexCoord);

  // Sepia Filter
  float sepiaR = 0.393*textureColor.r + 0.769*textureColor.g + 0.189*textureColor.b;
  float sepiaG = 0.349*textureColor.r + 0.686*textureColor.g + 0.168*textureColor.b;
  float sepiaB = 0.272*textureColor.r + 0.534*textureColor.g + 0.131*textureColor.b;
  vec4 sepiaColor = vec4(sepiaR, sepiaG, sepiaB, 1.0);
  
  // Gray Scale
  vec3 grayScale = vec3(0.5, 0.5, 0.5);
  vec4 grayColorFinal =  vec4( vec3(dot( textureColor.rgb, grayScale)), textureColor.a);

  if(uFlag == 1.0)
    fragColor = sepiaColor;
  else if(uFlag == 2.0)
    fragColor = grayColorFinal;
}`;

const vertexShaderCode4 = `#version 300 es
in vec3 aPosition;
in vec2 aTexCoords;

uniform mat4 uMMatrix;

out vec2 fragTexCoord;

void main() {
  // pass texture coordinate to frag shader
  fragTexCoord = aTexCoords;

  // calculate clip space position
  gl_Position =  uMMatrix * vec4(aPosition,1.0);
  gl_PointSize=3.0;
}`;

const fragShaderCode4 = `#version 300 es
precision highp float;

in vec2 fragTexCoord;

uniform sampler2D imageTexture;
uniform float uFlag; 
uniform float uContrast; 
uniform float uBrightness; 

out vec4 fragColor;

void main() {
  fragColor = vec4(0,0,0,1);
  
  // look up texture color
  vec4 textureColor =  texture(imageTexture, fragTexCoord);

  // Contrast
  vec3 contrastColor = 0.5+((uContrast+1.0)*(textureColor.rgb-0.5));
  vec4 contrastColorFinal = vec4(contrastColor, textureColor.a);

  // fragColor = contrastColorFinal + vec4(uBrightness);
  fragColor = contrastColorFinal+uBrightness;
}`;

const vertexShaderCode5 = `#version 300 es
in vec3 aPosition;
in vec2 aTexCoords;

uniform mat4 uMMatrix;

out vec2 fragTexCoord;

void main() {
  // pass texture coordinate to frag shader
  fragTexCoord = aTexCoords;

  // calculate clip space position
  gl_Position =  uMMatrix * vec4(aPosition,1.0);
  gl_PointSize=3.0;
}`;

const fragShaderCode5 = `#version 300 es
precision mediump float;

in vec2 fragTexCoord;

uniform vec4 diffuseTerm;
uniform sampler2D imageTexture;
uniform float uFlag;

const mat3 kernel1 =
mat3(
    1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0);

const mat3 kernel2 =
mat3(
    0.0, -1.0, 0.0,
    -1.0, 5.0, -1.0,
    0.0, -1.0, 0.0);

const mat3 kernel3 =
mat3(
    0.0, -1.0, 0.0,
    -1.0, 4.0, -1.0,
    0.0, -1.0, 0.0);

const mat3 kernel4 =
mat3(
    -3.0, -1.0, 0.0,
    -1.0, 5.0, 1.0,
    0.0, 1.0, 3.0);

out vec4 fragColor;

void main() {
  fragColor = vec4(0,0,0,1);
  vec4 colorSum = vec4(0,0,0,1);

  if(uFlag == 1.0){
    vec2 onePixel = vec2(float(1), float(1)) / vec2(400.0, 400.0);
  
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            colorSum += (texture(imageTexture, fragTexCoord + vec2(float(i), float(j)) * onePixel) * kernel1[i + 1][j + 1]);
        }
    }
  }
  else if(uFlag == 2.0){
    vec2 onePixel = vec2(float(1), float(1)) / vec2(400.0, 400.0); 
  
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            colorSum += (texture(imageTexture, fragTexCoord + vec2(float(i), float(j)) * onePixel) * kernel2[i + 1][j + 1]);
        }
    }
  }
  else if(uFlag == 3.0){
    // Sample the central pixel
    vec4 center = texture(imageTexture, fragTexCoord);

    // Sample the surrounding pixels
    vec4 top = texture(imageTexture, fragTexCoord + vec2(0.0, 1.0/400.0));
    vec4 bottom = texture(imageTexture, fragTexCoord + vec2(0.0, -1.0/400.0));
    vec4 left = texture(imageTexture, fragTexCoord + vec2(-1.0/400.0, 0.0));
    vec4 right = texture(imageTexture, fragTexCoord + vec2(1.0/400.0, 0.0));

    // Calculate the gradients (dx and dy)
    vec4 dx = (right - left);
    vec4 dy = (bottom - top);

    // Calculate the gradient magnitude
    vec4 gradientMagnitude = sqrt(dx * dx + dy * dy);

    colorSum += gradientMagnitude;
  }
  else if(uFlag == 4.0){
    vec2 onePixel = vec2(float(1), float(1)) / vec2(400.0, 400.0);
  
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            colorSum += (texture(imageTexture, fragTexCoord + vec2(float(i), float(j)) * onePixel) * kernel3[i + 1][j + 1]);
        }
    }
  }

  fragColor = colorSum;
}`;

function vertexShaderSetup(vertexShaderCode) {
  shader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(shader, vertexShaderCode);
  gl.compileShader(shader);
  // Error check whether the shader is compiled correctly
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    alert(gl.getShaderInfoLog(shader));
    return null;
  }
  return shader;
}

function fragmentShaderSetup(fragShaderCode) {
  shader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(shader, fragShaderCode);
  gl.compileShader(shader);
  // Error check whether the shader is compiled correctly
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    alert(gl.getShaderInfoLog(shader));
    return null;
  }
  return shader;
}

function initShaders(vertexShaderCode, fragShaderCode) {
  shaderProgram = gl.createProgram();

  var vertexShader = vertexShaderSetup(vertexShaderCode);
  var fragmentShader = fragmentShaderSetup(fragShaderCode);

  // attach the shaders
  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  //link the shader program
  gl.linkProgram(shaderProgram);

  // check for compiiion and linking status
  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    console.log(gl.getShaderInfoLog(vertexShader));
    console.log(gl.getShaderInfoLog(fragmentShader));
  }

  //finally use the program.
  gl.useProgram(shaderProgram);

  return shaderProgram;
}

function initGL(canvas) {
  try {
    gl = canvas.getContext("webgl2", {preserveDrawingBuffer: true}); // the graphics webgl2 context
    gl.viewportWidth = canvas.width; // the width of the canvas
    gl.viewportHeight = canvas.height; // the height
  } catch (e) {}
  if (!gl) {
    alert("WebGL initialization failed");
  }
}

function degToRad(degrees) {
  return (degrees * Math.PI) / 180;
}

function pushMatrix(stack, m) {
  //necessary because javascript only does shallow push
  var copy = mat4.create(m);
  stack.push(copy);
}

function popMatrix(stack) {
  if (stack.length > 0) return stack.pop();
  else console.log("stack has no matrix to pop!");
}

function initTextures(textureFile) {
  var tex = gl.createTexture();
  tex.image = new Image();
  tex.image.src = textureFile;
  tex.image.onload = function () {
    handleTextureLoaded(tex);
  };
  return tex;
}

function handleTextureLoaded(texture) {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1); // use it to flip Y if needed
  gl.texImage2D(
    gl.TEXTURE_2D, // 2D texture
    0, // mipmap level
    gl.RGBA, // internal format
    gl.RGBA, // format
    gl.UNSIGNED_BYTE, // type of data
    texture.image // array or <img>
  );

  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(
    gl.TEXTURE_2D,
    gl.TEXTURE_MIN_FILTER,
    gl.LINEAR_MIPMAP_LINEAR
  );

  drawScene();
}

// New square initialization function
function initSquare(){
  sqVertices = [
    0.5,  0.5,  0,
    -0.5,  0.5,  0, 
    - 0.5, -0.5, 0,
    0.5, -0.5,  0,
    ];
  sqIndices = [0,1,2, 0,2,3];    
  sqTexCoords = [1.0,1.0, 0.0,1.0, 0.0,0.0, 1.0,0.0]; 
}

function initSquareBuffer(){
  initSquare(); 
  sqVerPosBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, sqVerPosBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sqVertices), gl.STATIC_DRAW);
  sqVerPosBuf.itemSize = 3;
  sqVerPosBuf.numItems = 4;

  sqVerTexCoordsBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, sqVerTexCoordsBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sqTexCoords), gl.STATIC_DRAW);
  sqVerTexCoordsBuf.itemSize = 2;
  sqVerTexCoordsBuf.numItems = 4; 

  sqVerIndexBuf = gl.createBuffer();	
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sqVerIndexBuf); 
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(sqIndices), gl.STATIC_DRAW);  
  sqVerIndexBuf.itemsize = 1;
  sqVerIndexBuf.numItems = 6;  
}

function drawSquare(color, flag, contrast, brightness){
  gl.bindBuffer(gl.ARRAY_BUFFER, sqVerPosBuf);
  gl.vertexAttribPointer(
    aPositionLocation,
    sqVerPosBuf.itemSize,
    gl.FLOAT,
    false,
    0,
    0
  );

  // gl.bindBuffer(gl.ARRAY_BUFFER, sqVerNormalBuf);
  // gl.vertexAttribPointer(
  //   aNormalLocation,
  //   sqVerNormalBuf.itemSize,
  //   gl.FLOAT,
  //   false,
  //   0,
  //   0
  // );

  gl.bindBuffer(gl.ARRAY_BUFFER, sqVerTexCoordsBuf);
  gl.vertexAttribPointer(
    aTexCoordLocation,
    sqVerTexCoordsBuf.itemSize,
    gl.FLOAT,
    false,
    0,
    0
  );

  // Draw elementary arrays - triangle indices
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sqVerIndexBuf);
	gl.uniform1f(uFlagLocation, flag[0]);
  gl.uniform1f(uContrastLocation, contrast[0]);
	gl.uniform1f(uBrightLocation, brightness[0]);
  gl.uniform4fv(uDiffuseTermLocation, color);
  gl.uniformMatrix4fv(uMMatrixLocation, false, mMatrix);

  gl.drawElements(gl.TRIANGLES, sqVerIndexBuf.numItems, gl.UNSIGNED_SHORT, 0);
}

function initAlphaFBO(){
  // create a 2D texture in which framebuffer rendering will be stored
  alphaTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, alphaTexture);
  gl.texImage2D(
    gl.TEXTURE_2D, // target
    0, // mipmap level
    gl.RGBA, // internal format
    gl.viewportWidth, // width
    gl.viewportHeight, // height
    0, // border
    gl.RGBA, // format
    gl.UNSIGNED_BYTE, // type
    null // data, currently empty
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  // create an FBO
  var frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  // FBO.width = gl.viewportWidth;
  // FBO.height = gl.viewportHeight;

  // attach texture FBOtexture to the framebuffer FBO
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, alphaTexture, 0);

  // check FBO status
  var FBOstatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if(FBOstatus != gl.FRAMEBUFFER_COMPLETE)
    console.error("GL_FRAMEBUFFER_COMPLETE failed, CANNOT use FBO");

  gl.bindTexture(gl.TEXTURE_2D, null);
  return frameBuffer;
}

//////////////////////////////////////////////////////////////////////
//The main drawing routine
function drawScene() {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
  gl.clearColor(0.9, 0.9, 0.9, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  shaderProgram = initShaders(vertexShaderCode2, fragShaderCode2);

  //get locations of attributes declared in the vertex shader
  aPositionLocation = gl.getAttribLocation(shaderProgram, "aPosition");
  aTexCoordLocation = gl.getAttribLocation(shaderProgram, "aTexCoords");

  uMMatrixLocation = gl.getUniformLocation(shaderProgram, "uMMatrix");

  uDiffuseTermLocation = gl.getUniformLocation(shaderProgram, "diffuseTerm");
  uFlagLocation = gl.getUniformLocation(shaderProgram, "uFlag");
  uBrightLocation = gl.getUniformLocation(shaderProgram, "uBrightness");
  uContrastLocation = gl.getUniformLocation(shaderProgram, "uContrast");

  uTextureLocation = gl.getUniformLocation(shaderProgram, "imageTexture");

  //enable the attribute arrays
  gl.enableVertexAttribArray(aPositionLocation);
  gl.enableVertexAttribArray(aTexCoordLocation);

  //set up the model matrix
  mat4.identity(mMatrix);

  // back side
  pushMatrix(matrixStack, mMatrix);
  // transformations
  mMatrix = mat4.scale(mMatrix, [2,2,1]);
  drawSquare([0.0,1.0,1.0,1.0], [0.0], [0.0], [0.0]);
  mMatrix = popMatrix(matrixStack);

}

function loadImage2(){
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
  gl.clearColor(0.9, 0.9, 0.9, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  shaderProgram = initShaders(vertexShaderCode1, fragShaderCode1);

  //get locations of attributes declared in the vertex shader
  aPositionLocation = gl.getAttribLocation(shaderProgram, "aPosition");
  aTexCoordLocation = gl.getAttribLocation(shaderProgram, "aTexCoords");

  uMMatrixLocation = gl.getUniformLocation(shaderProgram, "uMMatrix");

  uDiffuseTermLocation = gl.getUniformLocation(shaderProgram, "diffuseTerm");
  uFlagLocation = gl.getUniformLocation(shaderProgram, "uFlag");
  uBrightLocation = gl.getUniformLocation(shaderProgram, "uBrightness");
  uContrastLocation = gl.getUniformLocation(shaderProgram, "uContrast");

  uTextureLocation = gl.getUniformLocation(shaderProgram, "imageTexture");

  //enable the attribute arrays
  gl.enableVertexAttribArray(aPositionLocation);
  gl.enableVertexAttribArray(aTexCoordLocation);

  //set up the model matrix
  mat4.identity(mMatrix);

  // back side
  pushMatrix(matrixStack, mMatrix);
  // texture setup for use
  gl.activeTexture(gl.TEXTURE0); // set texture unit 0 to use
  gl.bindTexture(gl.TEXTURE_2D, texture2); // bind the texture object
  gl.uniform1i(uTextureLocation, 0); // pass the texture unit
  // transformations
  mMatrix = mat4.translate(mMatrix, [0, 0, 0]);
  mMatrix = mat4.scale(mMatrix, [2,2,1]);
  drawSquare([0.0,1.0,1.0,1.0], [0.0], [0.0], [0.0]);
  mMatrix = popMatrix(matrixStack);
}

function loadImage1(){
  gl.bindFramebuffer(gl.FRAMEBUFFER, alphaFBO);
  gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
  gl.clearColor(0.9, 0.9, 0.9, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  shaderProgram = initShaders(vertexShaderCode, fragShaderCode);

  //get locations of attributes declared in the vertex shader
  aPositionLocation = gl.getAttribLocation(shaderProgram, "aPosition");
  aTexCoordLocation = gl.getAttribLocation(shaderProgram, "aTexCoords");

  uMMatrixLocation = gl.getUniformLocation(shaderProgram, "uMMatrix");

  uDiffuseTermLocation = gl.getUniformLocation(shaderProgram, "diffuseTerm");
  uFlagLocation = gl.getUniformLocation(shaderProgram, "uFlag");
  uBrightLocation = gl.getUniformLocation(shaderProgram, "uBrightness");
  uContrastLocation = gl.getUniformLocation(shaderProgram, "uContrast");

  uTextureLocation = gl.getUniformLocation(shaderProgram, "imageTexture");
  uTexture2Location = gl.getUniformLocation(shaderProgram, "imageTexture2");

  //enable the attribute arrays
  gl.enableVertexAttribArray(aPositionLocation);
  gl.enableVertexAttribArray(aTexCoordLocation);

  //set up the model matrix
  mat4.identity(mMatrix);

  // back side
  pushMatrix(matrixStack, mMatrix);
  // texture setup for use
  gl.activeTexture(gl.TEXTURE0); // set texture unit 0 to use
  gl.bindTexture(gl.TEXTURE_2D, texture1); // bind the texture object
  gl.uniform1i(uTextureLocation, 0); // pass the texture unit
  // texture setup for use
  gl.activeTexture(gl.TEXTURE1); // set texture unit 0 to use
  gl.bindTexture(gl.TEXTURE_2D, texture2); // bind the texture object
  gl.uniform1i(uTexture2Location, 1); // pass the texture unit
  // transformations
  mMatrix = mat4.translate(mMatrix, [0, 0, 0]);
  mMatrix = mat4.scale(mMatrix, [2,2,1]);
  drawSquare([0.0,1.0,1.0,1.0], [0.0], [0.0], [0.0]);
  mMatrix = popMatrix(matrixStack);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
  gl.clearColor(0.9, 0.9, 0.9, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  shaderProgram = initShaders(vertexShaderCode1, fragShaderCode1);

  //get locations of attributes declared in the vertex shader
  aPositionLocation = gl.getAttribLocation(shaderProgram, "aPosition");
  aTexCoordLocation = gl.getAttribLocation(shaderProgram, "aTexCoords");

  uMMatrixLocation = gl.getUniformLocation(shaderProgram, "uMMatrix");

  uDiffuseTermLocation = gl.getUniformLocation(shaderProgram, "diffuseTerm");
  uFlagLocation = gl.getUniformLocation(shaderProgram, "uFlag");
  uBrightLocation = gl.getUniformLocation(shaderProgram, "uBrightness");
  uContrastLocation = gl.getUniformLocation(shaderProgram, "uContrast");

  uTextureLocation = gl.getUniformLocation(shaderProgram, "imageTexture");

  //enable the attribute arrays
  gl.enableVertexAttribArray(aPositionLocation);
  gl.enableVertexAttribArray(aTexCoordLocation);

  //set up the model matrix
  mat4.identity(mMatrix);

  // back side
  pushMatrix(matrixStack, mMatrix);
  // texture setup for use
  gl.activeTexture(gl.TEXTURE0); // set texture unit 0 to use
  gl.bindTexture(gl.TEXTURE_2D, alphaTexture); // bind the texture object
  gl.uniform1i(uTextureLocation, 0); // pass the texture unit
  // transformations
  mMatrix = mat4.translate(mMatrix, [0, 0, 0]);
  mMatrix = mat4.scale(mMatrix, [2,2,1]);
  drawSquare([0.0,1.0,1.0,1.0], [0.0], [0.0], [0.0]);
  mMatrix = popMatrix(matrixStack);
}

function updateCheckboxes(checkboxId) {
  const sepiaCheckbox = document.getElementById("sepiaCheckbox");
  const grayscaleCheckbox = document.getElementById("grayscaleCheckbox");

  if (checkboxId === "sepiaCheckbox") {
    grayscaleCheckbox.checked = false;
  } else if (checkboxId === "grayscaleCheckbox") {
    sepiaCheckbox.checked = false;
  }
}

function applyEffectOnImage(texture, flag){
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
  gl.clearColor(0.9, 0.9, 0.9, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  shaderProgram = initShaders(vertexShaderCode3, fragShaderCode3);

  //get locations of attributes declared in the vertex shader
  aPositionLocation = gl.getAttribLocation(shaderProgram, "aPosition");
  aTexCoordLocation = gl.getAttribLocation(shaderProgram, "aTexCoords");

  uMMatrixLocation = gl.getUniformLocation(shaderProgram, "uMMatrix");

  uDiffuseTermLocation = gl.getUniformLocation(shaderProgram, "diffuseTerm");
  uFlagLocation = gl.getUniformLocation(shaderProgram, "uFlag");
  uBrightLocation = gl.getUniformLocation(shaderProgram, "uBrightness");
  uContrastLocation = gl.getUniformLocation(shaderProgram, "uContrast");

  uTextureLocation = gl.getUniformLocation(shaderProgram, "imageTexture");

  //enable the attribute arrays
  gl.enableVertexAttribArray(aPositionLocation);
  gl.enableVertexAttribArray(aTexCoordLocation);

  //set up the model matrix
  mat4.identity(mMatrix);

  // back side
  pushMatrix(matrixStack, mMatrix);
  // texture setup for use
  gl.activeTexture(gl.TEXTURE0); // set texture unit 0 to use
  gl.bindTexture(gl.TEXTURE_2D, texture); // bind the texture object
  gl.uniform1i(uTextureLocation, 0); // pass the texture unit
  // transformations
  mMatrix = mat4.translate(mMatrix, [0, 0, 0]);
  mMatrix = mat4.scale(mMatrix, [2,2,1]);
  drawSquare([0.0,1.0,1.0,1.0], flag, [0.0], [0.0]);
  mMatrix = popMatrix(matrixStack);
}

function applyEffects() {
  applySepia = document.getElementById("sepiaCheckbox").checked;
  applyGrayscale = document.getElementById("grayscaleCheckbox").checked;
  if(applySepia){
    if(flagtexture == 0)
      applyEffectOnImage(texture2, [1.0]);
    else
      applyEffectOnImage(alphaTexture, [1.0]);
  }
  else if(applyGrayscale){
    if(flagtexture == 0)
      applyEffectOnImage(texture2, [2.0]);
    else
      applyEffectOnImage(alphaTexture, [2.0]);
  }
  else{
    resetImage();
  }
}

function updateProcessCheckboxes(checkboxId) {
  const smoothCheckbox = document.getElementById("smoothCheckbox");
  const sharpenCheckbox = document.getElementById("sharpenCheckbox");
  const gradientCheckbox = document.getElementById("gradientCheckbox");
  const laplacianCheckbox = document.getElementById("laplacianCheckbox");

  if (checkboxId === "smoothCheckbox") {
    sharpenCheckbox.checked = false;
    gradientCheckbox.checked = false;
    laplacianCheckbox.checked = false;
  } 
  else if (checkboxId === "sharpenCheckbox") {
    smoothCheckbox.checked = false;
    gradientCheckbox.checked = false;
    laplacianCheckbox.checked = false;
  }
  else if (checkboxId === "gradientCheckbox") {
    smoothCheckbox.checked = false;
    sharpenCheckbox.checked = false;
    laplacianCheckbox.checked = false;
  }
  else if (checkboxId === "laplacianCheckbox") {
    smoothCheckbox.checked = false;
    gradientCheckbox.checked = false;
    gradientCheckbox.checked = false;
  }
}

function applyProcessEffectOnImage(texture, flag){
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
  gl.clearColor(0.9, 0.9, 0.9, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  shaderProgram = initShaders(vertexShaderCode5, fragShaderCode5);

  //get locations of attributes declared in the vertex shader
  aPositionLocation = gl.getAttribLocation(shaderProgram, "aPosition");
  aTexCoordLocation = gl.getAttribLocation(shaderProgram, "aTexCoords");

  uMMatrixLocation = gl.getUniformLocation(shaderProgram, "uMMatrix");

  uDiffuseTermLocation = gl.getUniformLocation(shaderProgram, "diffuseTerm");
  uFlagLocation = gl.getUniformLocation(shaderProgram, "uFlag");
  uBrightLocation = gl.getUniformLocation(shaderProgram, "uBrightness");
  uContrastLocation = gl.getUniformLocation(shaderProgram, "uContrast");

  uTextureLocation = gl.getUniformLocation(shaderProgram, "imageTexture");

  //enable the attribute arrays
  gl.enableVertexAttribArray(aPositionLocation);
  gl.enableVertexAttribArray(aTexCoordLocation);

  //set up the model matrix
  mat4.identity(mMatrix);

  // back side
  pushMatrix(matrixStack, mMatrix);
  // texture setup for use
  gl.activeTexture(gl.TEXTURE0); // set texture unit 0 to use
  gl.bindTexture(gl.TEXTURE_2D, texture); // bind the texture object
  gl.uniform1i(uTextureLocation, 0); // pass the texture unit
  // transformations
  mMatrix = mat4.translate(mMatrix, [0, 0, 0]);
  mMatrix = mat4.scale(mMatrix, [2,2,1]);
  drawSquare([0.0,1.0,1.0,1.0], flag, [0.0], [0.0]);
  mMatrix = popMatrix(matrixStack);
}

function applyProcessEffects(){
  applySmooth = document.getElementById("smoothCheckbox").checked;
  applySharpen = document.getElementById("sharpenCheckbox").checked;
  applyGradient = document.getElementById("gradientCheckbox").checked;
  applyLaplacian = document.getElementById("laplacianCheckbox").checked;
  if(applySmooth){
    if(flagtexture == 0)
      applyProcessEffectOnImage(texture2, [1.0]);
    else
      applyProcessEffectOnImage(alphaTexture, [1.0]);
  }
  else if(applySharpen){
    if(flagtexture == 0)
      applyProcessEffectOnImage(texture2, [2.0]);
    else
      applyProcessEffectOnImage(alphaTexture, [2.0]);
  }
  else if(applyGradient){
    if(flagtexture == 0)
      applyProcessEffectOnImage(texture2, [3.0]);
    else
      applyProcessEffectOnImage(alphaTexture, [3.0]);
  }
  else if(applyLaplacian){
    if(flagtexture == 0)
      applyProcessEffectOnImage(texture2, [4.0]);
    else
      applyProcessEffectOnImage(alphaTexture, [4.0]);
  }
  else{
    resetImage();
  }
}

function applyEffectOnImageCB(texture, flag, contrast, brightness){
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
  gl.clearColor(0.9, 0.9, 0.9, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  shaderProgram = initShaders(vertexShaderCode4, fragShaderCode4);

  //get locations of attributes declared in the vertex shader
  aPositionLocation = gl.getAttribLocation(shaderProgram, "aPosition");
  aTexCoordLocation = gl.getAttribLocation(shaderProgram, "aTexCoords");

  uMMatrixLocation = gl.getUniformLocation(shaderProgram, "uMMatrix");

  uDiffuseTermLocation = gl.getUniformLocation(shaderProgram, "diffuseTerm");
  uFlagLocation = gl.getUniformLocation(shaderProgram, "uFlag");
  uBrightLocation = gl.getUniformLocation(shaderProgram, "uBrightness");
  uContrastLocation = gl.getUniformLocation(shaderProgram, "uContrast");

  uTextureLocation = gl.getUniformLocation(shaderProgram, "imageTexture");

  //enable the attribute arrays
  gl.enableVertexAttribArray(aPositionLocation);
  gl.enableVertexAttribArray(aTexCoordLocation);

  //set up the model matrix
  mat4.identity(mMatrix);

  // back side
  pushMatrix(matrixStack, mMatrix);
  // texture setup for use
  gl.activeTexture(gl.TEXTURE0); // set texture unit 0 to use
  gl.bindTexture(gl.TEXTURE_2D, texture); // bind the texture object
  gl.uniform1i(uTextureLocation, 0); // pass the texture unit
  // transformations
  mMatrix = mat4.translate(mMatrix, [0, 0, 0]);
  mMatrix = mat4.scale(mMatrix, [2,2,1]);
  drawSquare([0.0,1.0,1.0,1.0], flag, contrast, brightness);
  mMatrix = popMatrix(matrixStack);
}

function applyCBEffects(){
  if(flagtexture == 0){
    applyEffectOnImageCB(texture2, [0.0], contrast, brightness);
  }
  else{
    applyEffectOnImageCB(alphaTexture, [0.0], contrast, brightness);
  }
}

function selectImageMode() {
  const radios = document.getElementsByName("imageType");
  for (const radio of radios) {
    if (radio.checked) {
      selectedImageType = radio.value;
      if (selectedImageType === "background") {
        loadImage2();
        flagtexture = 0;
      } else if (selectedImageType === "alphablended") {
        loadImage1();
        flagtexture = 1;
      }
    }
  }
}

function changeContrast(event){
  contrast[0] = event.target.value;
  applyCBEffects();
}

function changeBrightness(event){
  brightness[0] = event.target.value;
  applyCBEffects();
}

function resetImage() {
  if(flagtexture == 0){
    loadImage2();
  }
  else{
    loadImage1();
  }
}

function resetImage2() {
  loadImage2();
}

// This is the entry point from the html
function webGLStart() {
  canvas = document.getElementById("assignment3");

  // Load foreground image
  foregroundImageInput = document.getElementById('foregroundImageInput');
  foregroundImageInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
  
    reader.onload = (e) => {
      texture1 = gl.createTexture();
      texture1.image = new Image();
      texture1.image.src = e.target.result;
      texture1.image.onload = function () {
        handleTextureLoaded(texture1);
      };
    };
  
    reader.readAsDataURL(file);
  });

  // Load background image
  backgroundImageInput = document.getElementById('backgroundImageInput');
  backgroundImageInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
  
    reader.onload = (e) => {
      texture2 = gl.createTexture();
      texture2.image = new Image();
      texture2.image.src = e.target.result;
      texture2.image.onload = function () {
        handleTextureLoaded(texture2);
      };
    };
  
    reader.readAsDataURL(file);
  });

  document.getElementById("contrastSlider").oninput=changeContrast;
  document.getElementById("brightnessSlider").oninput=changeBrightness;
  document.getElementById("resetButton").addEventListener("click", resetImage2);
  const resetButton = document.getElementById("resetButton");
  resetButton.onclick = function(){
    document.getElementById('contrastSlider').value = 0;
    document.getElementById('brightnessSlider').value = 0;
  };


  // Get a reference to the "Screenshot" button element
  const screenshotButton = document.getElementById("screenshotButton");
  screenshotButton.addEventListener('click', () => {
    canvas.toBlob((blob) => {
      saveBlob(blob, `screencapture-${canvas.width}x${canvas.height}.png`);
    });
  });
   
  const saveBlob = (function() {
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style.display = 'none';
    return function saveData(blob, fileName) {
       const url = window.URL.createObjectURL(blob);
       a.href = url;
       a.download = fileName;
       a.click();
    };
  }());

  const radioInputs = document.getElementsByName("imageType");
  for (const radio of radioInputs) {
    radio.addEventListener("change", selectImageMode);
  }

  const sepiaCheckbox = document.getElementById("sepiaCheckbox");
  sepiaCheckbox.addEventListener("change", () => {
    updateCheckboxes("sepiaCheckbox");
    applyEffects();
  });

  const grayscaleCheckbox = document.getElementById("grayscaleCheckbox");
  grayscaleCheckbox.addEventListener("change", () => {
    updateCheckboxes("grayscaleCheckbox");
    applyEffects();
  });

  const smoothCheckbox = document.getElementById("smoothCheckbox");
  smoothCheckbox.addEventListener("change", () => {
    updateProcessCheckboxes("smoothCheckbox");
    applyProcessEffects();
  });

  const sharpenCheckbox = document.getElementById("sharpenCheckbox");
  sharpenCheckbox.addEventListener("change", () => {
    updateProcessCheckboxes("sharpenCheckbox");
    applyProcessEffects();
  });

  const gradientCheckbox = document.getElementById("gradientCheckbox");
  gradientCheckbox.addEventListener("change", () => {
    updateProcessCheckboxes("gradientCheckbox");
    applyProcessEffects();
  });

  const laplacianCheckbox = document.getElementById("laplacianCheckbox");
  laplacianCheckbox.addEventListener("change", () => {
    updateProcessCheckboxes("laplacianCheckbox");
    applyProcessEffects();
  });

  initGL(canvas);

  //initialize buffers for the square
  initSquareBuffer();

  alphaFBO = initAlphaFBO();
  
  drawScene();
}
