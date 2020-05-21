//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Váradi Richárd Tamás
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders

class Circle{
private:
    unsigned int vao;
    const int nv = 100;
public:
    void buffer(){
        glGenVertexArrays(1, &vao);
    }

    //Szirmay Tanár Úr videója alapján https://www.youtube.com/watch?v=uPedQt5Tcpk
    void drawCirlce(){
        vec2 vertices[nv];
        for (int i = 0; i < nv; i++){
            float fi = i * 2 * M_PI / nv;
            vertices[i] = vec2(cosf(fi), sinf(fi));
        }
        int location = glGetUniformLocation(gpuProgram.getId(), "color");
        glUniform3f(location, 0.45f, 0.5f, 0.49f); // 3 floats

        glBindVertexArray(vao);
        unsigned int vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * nv, vertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

        glBindVertexArray(vao);

        glDrawArrays(GL_TRIANGLE_FAN, 0, nv);
    }
}circle;

class Points{
private:
    unsigned int vao;
public:
    std::vector<vec2> listOfPoints;
    int cntPoints = 0;
    void buffer(){
        glGenVertexArrays(1, &vao);
    }

    void addPoint(float x, float y){
        if (cntPoints < 3){
            listOfPoints.push_back(vec2(x, y));
            cntPoints++;
        }
    }

    void drawPoints(){
        int location = glGetUniformLocation(gpuProgram.getId(), "color");
        glUniform3f(location, 0.1058f, 0.2313f, 0.5804f); // 3 floats

        glBindVertexArray(vao);
        unsigned int vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * listOfPoints.size(), &*(listOfPoints.begin()), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

        glBindVertexArray(vao);

        glPointSize(10);

        glDrawArrays(GL_POINTS, 0, listOfPoints.size());
    }
}p;

class Triangle{
private:
    unsigned int vao;
public:
    std::vector<vec2> trianglesPoints;
    void buffer(){
        glGenVertexArrays(1, &vao);
    }

    //Házi feladat kiadásánál tartott bemutató és a sziriusz.pptx alapján számoltam ki lapon paraméteresen és ide helyettesítettem be.
    float calcLine(vec2 p, vec2 q, std::vector<vec2>& list1, std::vector<vec2>& list2){
        float cy = (p.x + p.x * (q.x * q.x) - q.x - q.x * (p.y * p.y) - q.x * (p.x * p.x) + (q.y * q.y) * p.x) / (2 * (q.y * p.x - q.x * p.y ));
        float cx = 1/(2 * p.x) + p.x / 2 + (p.y * p.y) / (2 * p.x) - (p.y * cy) / p.x;
        vec2 c(cx, cy);
        list1.push_back(c);
        list2.push_back(c);
        float rad = length(p - c);

        float p1Rad, p2Rad;
        if(atan2(p.y - cy, p.x-cx) < atan2(q.y - cy, q.x-cx)){
            p1Rad = atan2(p.y - cy, p.x-cx);
            p2Rad = atan2(q.y - cy, q.x-cx);
        } else {
            p1Rad = atan2(q.y - cy, q.x-cx);
            p2Rad = atan2(p.y - cy, p.x-cx);
        }

        if((p2Rad - p1Rad) > M_PI){
            float tmp = p2Rad;
            p2Rad = p1Rad + 2 * M_PI;
            p1Rad = tmp;
        }

        float ds = 0;
        vec2 tmp;
        if (p.x < 0 && p.y > 0 && q.x > 0 && q.y < 0){
            tmp = q;
        } else if (q.x < 0 && q.y > 0 && p.x > 0 && p.y < 0){
            tmp = q;
        } else {
            tmp = p;
        }
        for (float deg = p1Rad; deg < p2Rad; deg += (0.36 * M_PI/180)) {
            vec2 newPoint(rad * cos(deg) + cx, rad * sin(deg) + cy);
            trianglesPoints.push_back(newPoint);
            vec2 diff = newPoint - tmp;
            ds += length(diff) / (1.0f - tmp.x*tmp.x - tmp.y*tmp.y);
            tmp = newPoint;
        }
        return ds;
    }

    void drawTriangle(){
        int location = glGetUniformLocation(gpuProgram.getId(), "color");
        glUniform3f(location, 0.1058f, 0.2313f, 0.5804f); // 3 floats

        glBindVertexArray(vao);
        unsigned int vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * trianglesPoints.size(), &*(trianglesPoints.begin()), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

        glBindVertexArray(vao);

        glLineWidth(5);

        glDrawArrays(GL_LINE_LOOP, 0, trianglesPoints.size());
    }
}triangle;

bool operator==(const vec2& v1, const vec2& v2){
    if(fabs(v1.x-v2.x) < 0.0005 && fabs(v1.y - v2.y) < 0.0005){
        return true;
    }
    return false;
}

class TriangleFill{
private:
    unsigned int vao;
public:
    void buffer(){
        glGenVertexArrays(1, &vao);
    }

    //Line intersect: https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
    bool inter(vec2 line_p1, vec2 line_p2, vec2 line_q1, vec2 line_q2){
        if(line_p1 == line_q1 || line_p1 == line_q2 ||
           line_p2 == line_q1 || line_p2 == line_q2)
            return false;

        float A = line_p1.x - line_p2.x;
        float B = line_q2.x - line_q1.x;
        float C = line_p1.y - line_p2.y;
        float D = line_q2.y - line_q1.y;
        float E = line_q2.x - line_p2.x;
        float F = line_q2.y - line_p2.y;

        float f_ce_pa = F-C * E/A;
        float d_cb_pa = D-C * B/A;
        float t2 = f_ce_pa / d_cb_pa;

        if(t2 > 1 || t2 < 0)
            return false;

        float t1 = E/A - B/A *((F-C*E/A)/(D-C*B/A));

        return (t1 > 0 && t1 < 1);
    }

    int index(int i, int sum){
        return (i >= 0 ? i % sum : sum - ((-i) % sum)) % sum;
    }

    bool ray(vec2 point, vec2 v1, vec2 v2) {
        if (point.y == v2.y)
            return false;

        if (v1.y > v2.y)
            std::swap(v1, v2);
        if ((point.y > v2.y) || (point.y < v1.y))
            return false;
        if (point.x > std::max(v1.x, v2.x))
            return false;
        if (point.x < std::min(v1.x, v2.x))
            return true;

        float slopeAB = (v2.y - v1.y) / (v2.x - v1.x);
        return ((slopeAB > 0 ?  ((point.y - v1.y) / (point.x - v1.x) > slopeAB) : ((v1.y - point.y) / (v1.x - point.x) > slopeAB)));
    }

    bool doesContain(const vec2& point, const std::vector<vec2>& polivertices){
        bool contain = false;
        if(ray(point, polivertices.back(), polivertices.front()))
            contain = !contain;

        for(int i = 0; i < polivertices.size() - 1; ++i){
            if(ray(point, polivertices[i], polivertices[i + 1])){
                contain = !contain;
            }
        }
        return contain;
    }

    bool isDiagonal(const vec2& point1, const vec2& point2, const std::vector<vec2>& polivertices){
        if(inter(point1, point2, polivertices.back(), polivertices.front())){
            return false;
        }

        for(size_t i=0;i<polivertices.size()-1;++i){
            if(inter(point1, point2, polivertices[i], polivertices[i+1])){
                return false;
            }
        }
        if(!doesContain((point1+point2)/2, polivertices)){
            return false;
        }

        return true;
    }

    void earclipping(std::vector<vec2>& triangles, std::vector<vec2> polivertices){
        bool simplePolynom = false;
        for (int i = 0; !simplePolynom && i < polivertices.size(); ++i) {
            if (isDiagonal(polivertices[index(i - 1, polivertices.size())], polivertices[index(i + 1, polivertices.size())], polivertices)) {
                triangles.push_back(polivertices[index(i - 1, polivertices.size())]);
                triangles.push_back(polivertices[i]);
                triangles.push_back(polivertices[index(i + 1, polivertices.size())]);
                polivertices.erase(polivertices.begin() + i);
                simplePolynom = true;
            }
        }
        if (simplePolynom) {
            earclipping(triangles, polivertices);
            return;
        }
    }

    void drawTriangleFill(std::vector<vec2> triangles){
        int location = glGetUniformLocation(gpuProgram.getId(), "color");
        glUniform3f(location, 0.588f, 0.0941f, 0.0588f); // 3 floats

        std::vector<vec2> vertices;
        earclipping(vertices,triangles);

        glBindVertexArray(vao);
        unsigned int vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * vertices.size(), &*(vertices.begin()), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

        glBindVertexArray(vao);

        glDrawArrays(GL_TRIANGLES, 0, vertices.size());
    }
}triangleFill;

//https://www.youtube.com/watch?v=TTom8n3FFCw
void calcAngles(std::vector<vec2> angleA, std::vector<vec2> angleB, std::vector<vec2> angleC){
    float A = acos((dot(angleA[1]-angleA[0], angleA[2]-angleA[0]))/ (length(angleA[1] - angleA[0]) * length(angleA[2] - angleA[0])));
    float B = acos((dot(angleB[1]-angleB[0], angleB[2]-angleB[0]))/ (length(angleB[1] - angleB[0]) * length(angleB[2] - angleB[0])));
    float C = acos((dot(angleC[1]-angleC[0], angleC[2]-angleC[0]))/ (length(angleC[1] - angleC[0]) * length(angleC[2] - angleC[0])));

    float degA = A * 180.0 / M_PI;
    degA = ((360-2*degA)/2);
    float degB = B * 180.0 / M_PI;
    degB = ((360-2*degB)/2);
    float degC = C * 180.0 / M_PI;
    degC = ((360-2*degC)/2);

    printf("Alpha: %f Beta: %f Gamma: %f Angle sum: %f\n", degA, degB, degC, degA + degB + degC);
}
// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    circle.buffer();
    p.buffer();
    triangle.buffer();
    triangleFill.buffer();

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

bool out = true;
std::vector<vec2> tmp;
// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);     // background color
    glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

    // Set color to (0, 1, 0) = green
    int location = glGetUniformLocation(gpuProgram.getId(), "color");
    glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

    float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix,
                              0, 1, 0, 0,    // row-major!
                              0, 0, 1, 0,
                              0, 0, 0, 1 };

    location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
    glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

    circle.drawCirlce();
    p.drawPoints();

    if (p.listOfPoints.size() == 3){
        std::vector<vec2> angleA; angleA.push_back(vec2(p.listOfPoints[0].x, p.listOfPoints[0].y));
        std::vector<vec2> angleB; angleB.push_back(vec2(p.listOfPoints[1].x, p.listOfPoints[1].y));
        std::vector<vec2> angleC; angleC.push_back(vec2(p.listOfPoints[2].x, p.listOfPoints[2].y));

        float a = triangle.calcLine(p.listOfPoints[0], p.listOfPoints[1], angleA, angleB);
        float b = triangle.calcLine(p.listOfPoints[1], p.listOfPoints[2], angleB, angleC);
        float c = triangle.calcLine(p.listOfPoints[0], p.listOfPoints[2], angleA, angleC);

        if(out){
            calcAngles(angleA, angleB, angleC);
            printf("a: %f b: %f c: %f\n", a, b, c);
        }

        if(tmp.size() == 0){
            tmp = triangle.trianglesPoints;
        }
        triangleFill.drawTriangleFill(tmp);
        triangle.drawTriangle();
        out = false;
    }

    glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    // Convert to normalized device space
    float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
    float cY = 1.0f - 2.0f * pY / windowHeight;
    printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    // Convert to normalized device space
    float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
    float cY = 1.0f - 2.0f * pY / windowHeight;

    char * buttonStat;
    switch (state) {
        case GLUT_DOWN: buttonStat = "pressed"; break;
        case GLUT_UP:   buttonStat = "released"; break;
    }

    switch (button) {
        case GLUT_LEFT_BUTTON:  // printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
            if (state == GLUT_DOWN){
                p.addPoint(cX, cY);
                glutPostRedisplay();
            }
            break;
        case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
        case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
    }
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
