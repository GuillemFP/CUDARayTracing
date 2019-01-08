#include "Color.h"

#include "Vector3.h"
#include <algorithm>

Color::Color(float r, float g, float b, float a) : r(r), g(g), b(b), a(a)
{
}

Color::Color(const Vector3& color) : r(color.x()), g(color.y()), b(color.z()), a(1.0f)
{

}

void Color::SetColor(float r, float g, float b)
{
	this->r = r;
	this->g = g;
	this->b = b;
}

Color& Color::Capped()
{
	r = std::max(0.0f, std::min(1.0f, r));
	g = std::max(0.0f, std::min(1.0f, g));
	b = std::max(0.0f, std::min(1.0f, b));
	return *this;
}
