// @ts-nocheck
// Express REST API Template with Error Handling 
import express, { Request, Response, NextFunction } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import { z } from 'zod';

// ============ Error Classes ============
export class AppError extends Error {
    constructor(
        message: string,
        public statusCode: number = 500,
        public code: string = 'INTERNAL_ERROR',
        public details?: any
    ) {
        super(message);
        // captureStackTrace is a V8/Node.js-specific API
        if (typeof (Error as any).captureStackTrace === 'function') {
            (Error as any).captureStackTrace(this, this.constructor);
        }
    }
}

export class ValidationError extends AppError {
    constructor(message: string, details?: any) {
        super(message, 400, 'VALIDATION_ERROR', details);
    }
}

export class NotFoundError extends AppError {
    constructor(resource: string, id: string) {
        super(`${resource} not found`, 404, 'NOT_FOUND', { resource, id });
    }
}

// ============ Validation Middleware ============
const validate = (schema: z.ZodSchema) => {
    return async (req: Request, res: Response, next: NextFunction) => {
        try {
            await schema.parseAsync({
                body: req.body,
                query: req.query,
                params: req.params,
            });
            next();
        } catch (error) {
            if (error instanceof z.ZodError) {
                next(new ValidationError('Validation failed', error.errors));
            } else {
                next(error);
            }
        }
    };
};

// ============ Error Handler ============
const errorHandler = (
    err: Error,
    req: Request,
    res: Response,
    next: NextFunction
) => {
    if (err instanceof AppError) {
        return res.status(err.statusCode).json({
            success: false,
            error: {
                code: err.code,
                message: err.message,
                details: err.details,
            },
        });
    }

    console.error('Unhandled error:', err);
    res.status(500).json({
        success: false,
        error: {
            code: 'INTERNAL_ERROR',
            message: 'Something went wrong',
        },
    });
};

// ============ App Setup ============
const app = express();

app.use(helmet());
app.use(cors({ origin: process.env.CORS_ORIGIN?.split(',') }));
app.use(express.json({ limit: '10mb' }));

// ============ Schemas ============
const createUserSchema = z.object({
    body: z.object({
        email: z.string().email(),
        name: z.string().min(1),
        password: z.string().min(8),
    }),
});

const getUserParamsSchema = z.object({
    params: z.object({
        id: z.string().uuid(),
    }),
});

// ============ Routes ============
app.get('/api/v1/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.post(
    '/api/v1/users',
    validate(createUserSchema),
    async (req: Request, res: Response, next: NextFunction) => {
        try {
            const { email, name, password } = req.body;

            // TODO: Check if user exists, hash password, save to DB
            const user = { id: 'generated-uuid', email, name, createdAt: new Date() };

            res.status(201).json({
                success: true,
                data: user,
            });
        } catch (error) {
            next(error);
        }
    }
);

app.get(
    '/api/v1/users/:id',
    validate(getUserParamsSchema),
    async (req: Request, res: Response, next: NextFunction) => {
        try {
            const { id } = req.params;

            // TODO: Fetch from database
            const user = null; // await db.user.findUnique({ where: { id } });

            if (!user) {
                throw new NotFoundError('User', id);
            }

            res.json({ success: true, data: user });
        } catch (error) {
            next(error);
        }
    }
);

// ============ Error Handler (must be last) ============
app.use(errorHandler);

// ============ Start Server ============
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

export { app };
